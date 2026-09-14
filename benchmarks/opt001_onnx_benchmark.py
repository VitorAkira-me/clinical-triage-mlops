"""OPT-001/BENCH-001 — converte o baseline (ML-003) para ONNX e compara com o
original sklearn: corretude primeiro, depois latência. Números reais, não estimados.

Uso:
    uv run --extra benchmark python benchmarks/opt001_onnx_benchmark.py

Escreve docs/experiments/OPT-001-benchmark.json com os resultados.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/tfidf_logreg_baseline.joblib")
RAW_PARQUET = Path("data/raw/fedmml_ed_triage_raw.parquet")
RESULTS_PATH = Path("docs/experiments/OPT-001-benchmark.json")

N_CORRECTNESS_SAMPLES = 150
N_LATENCY_ITERATIONS = 300
LATENCY_TEXT = (
    "67yo M c/o chest pain, moderate distress. Rapid assessment indicates emergent condition."
)


def load_sample_texts(n: int) -> list[str]:
    df = pd.read_parquet(RAW_PARQUET)
    df = df.dropna(subset=["clinical_notes"])
    return df["clinical_notes"].sample(n, random_state=7).tolist()


# --- Tentativa 1: pipeline INTEIRO (TfidfVectorizer + LogisticRegression) em ONNX ----------


def try_convert_full_pipeline(pipeline) -> bytes | None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType

    initial_type = [("input", StringTensorType([None, 1]))]
    try:
        onnx_model = convert_sklearn(
            pipeline, initial_types=initial_type, options={id(pipeline): {"zipmap": False}}
        )
        return onnx_model.SerializeToString()
    except Exception:
        logger.exception("Conversão do pipeline INTEIRO falhou")
        return None


def predict_full_onnx(
    session: rt.InferenceSession, texts: list[str]
) -> tuple[list[str], np.ndarray]:
    input_name = session.get_inputs()[0].name
    arr = np.array([[t] for t in texts], dtype=object)
    labels, proba = session.run(None, {input_name: arr})
    return list(labels), np.asarray(proba)


# --- Tentativa 2 (fallback): só o LogisticRegression em ONNX, TF-IDF continua em sklearn ----


def convert_classifier_only(pipeline) -> bytes:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    clf = pipeline.named_steps["clf"]
    n_features = len(pipeline.named_steps["tfidf"].vocabulary_)
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        clf, initial_types=initial_type, options={id(clf): {"zipmap": False}}
    )
    return onnx_model.SerializeToString()


def predict_classifier_only_onnx(
    pipeline, session: rt.InferenceSession, texts: list[str]
) -> tuple[list[str], np.ndarray]:
    input_name = session.get_inputs()[0].name
    x = pipeline.named_steps["tfidf"].transform(texts).toarray().astype(np.float32)
    labels, proba = session.run(None, {input_name: x})
    return list(labels), np.asarray(proba)


# --- Corretude ------------------------------------------------------------------------------


def check_correctness(pipeline, predict_onnx_fn, texts: list[str], classes: list[str]) -> dict:
    sk_labels = pipeline.predict(texts).tolist()
    sk_proba = pipeline.predict_proba(texts)  # colunas na ordem de pipeline.classes_

    onnx_labels, onnx_proba_raw = predict_onnx_fn(texts)

    # onnxruntime devolve proba como lista de dicts {classe: prob} quando zipmap=False só se
    # a saída já vier como tensor; alinhamos por pipeline.classes_ explicitamente, nunca por
    # posição assumida — mesmo cuidado do gotcha já documentado na API-001.
    if isinstance(onnx_proba_raw, np.ndarray) and onnx_proba_raw.dtype != object:
        onnx_proba = onnx_proba_raw
    else:
        # veio como array de dicts (zipmap) — remonta na ordem de `classes`
        onnx_proba = np.array([[d[c] for c in classes] for d in onnx_proba_raw])

    labels_match = sum(a == b for a, b in zip(sk_labels, onnx_labels, strict=True))
    max_abs_diff = float(np.max(np.abs(sk_proba - onnx_proba)))
    proba_close = bool(np.allclose(sk_proba, onnx_proba, atol=1e-4))

    return {
        "n_samples": len(texts),
        "labels_match": labels_match,
        "labels_match_pct": labels_match / len(texts),
        "max_abs_proba_diff": max_abs_diff,
        "probabilities_close_atol_1e-4": proba_close,
    }


# --- Latência --------------------------------------------------------------------------------


def benchmark_latency(fn, n: int) -> dict:
    # warmup
    for _ in range(10):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1000  # ms
    return {
        "n_iterations": n,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def main() -> None:
    pipeline = joblib.load(MODEL_PATH)
    classes = list(pipeline.classes_)
    logger.info("Modelo carregado: %s, classes=%s", MODEL_PATH, classes)

    texts = load_sample_texts(N_CORRECTNESS_SAMPLES)
    logger.info("Amostra de corretude: %d textos reais do dataset", len(texts))

    result: dict = {"classes": classes}

    logger.info("=== Tentativa 1: pipeline INTEIRO em ONNX ===")
    full_onnx_bytes = try_convert_full_pipeline(pipeline)

    if full_onnx_bytes is not None:
        session = rt.InferenceSession(full_onnx_bytes)
        correctness = check_correctness(
            pipeline, lambda t: predict_full_onnx(session, t), texts, classes
        )
        logger.info("Corretude (pipeline inteiro): %s", correctness)
        if correctness["labels_match_pct"] == 1.0 and correctness["probabilities_close_atol_1e-4"]:
            result["approach"] = "full_pipeline"
            result["correctness"] = correctness
            onnx_path = Path("models/tfidf_logreg_baseline.onnx")
            onnx_path.write_bytes(full_onnx_bytes)
            result["onnx_path"] = onnx_path.as_posix()

            def onnx_call():
                predict_full_onnx(session, [LATENCY_TEXT])

        else:
            logger.warning(
                "Pipeline inteiro converteu mas NÃO bateu corretude — descartando, indo pro fallback."
            )
            full_onnx_bytes = None

    if full_onnx_bytes is None:
        logger.info("=== Fallback: só o classificador (LogisticRegression) em ONNX ===")
        clf_onnx_bytes = convert_classifier_only(pipeline)
        session = rt.InferenceSession(clf_onnx_bytes)
        correctness = check_correctness(
            pipeline,
            lambda t: predict_classifier_only_onnx(pipeline, session, t),
            texts,
            classes,
        )
        logger.info("Corretude (classificador via ONNX, TF-IDF em sklearn): %s", correctness)
        if not (
            correctness["labels_match_pct"] == 1.0 and correctness["probabilities_close_atol_1e-4"]
        ):
            raise RuntimeError(
                f"Fallback também não bateu corretude: {correctness} — não seguir pra latência "
                "com um resultado errado."
            )
        result["approach"] = "classifier_only"
        result["correctness"] = correctness
        onnx_path = Path("models/tfidf_logreg_baseline_clf.onnx")
        onnx_path.write_bytes(clf_onnx_bytes)
        result["onnx_path"] = str(onnx_path)

        def onnx_call():
            predict_classifier_only_onnx(pipeline, session, [LATENCY_TEXT])

    logger.info(
        "=== Latência: sklearn original vs. ONNX (%d iterações cada) ===", N_LATENCY_ITERATIONS
    )

    def sklearn_call():
        pipeline.predict_proba([LATENCY_TEXT])

    sklearn_latency = benchmark_latency(sklearn_call, N_LATENCY_ITERATIONS)
    onnx_latency = benchmark_latency(onnx_call, N_LATENCY_ITERATIONS)

    result["latency_text"] = LATENCY_TEXT
    result["sklearn_original"] = sklearn_latency
    result["onnx"] = onnx_latency
    result["speedup_median"] = sklearn_latency["median_ms"] / onnx_latency["median_ms"]
    result["speedup_p95"] = sklearn_latency["p95_ms"] / onnx_latency["p95_ms"]

    logger.info("Abordagem final: %s", result["approach"])
    logger.info("sklearn: %s", sklearn_latency)
    logger.info("onnx:    %s", onnx_latency)
    logger.info(
        "speedup (mediana): %.2fx | speedup (p95): %.2fx",
        result["speedup_median"],
        result["speedup_p95"],
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Resultados salvos em %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
