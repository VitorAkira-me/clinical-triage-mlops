"""Gera um modelo placeholder para validar o build da imagem Docker (CI-001).

NÃO é o baseline real — esse vem de notebooks/02_baseline.ipynb (ML-003) e é gitignored.
Este script treina um Pipeline TF-IDF + LogisticRegression minúsculo sobre poucas linhas
sintéticas (mesmas classes e formato de tests/test_api.py), só para o Dockerfile ter um
`.joblib` válido no COPY — permitindo que o job de build (CI-001) e testes manuais locais
validem a imagem inteira (build + subir container + /health + /predict) sem depender do
notebook completo nem de um artefato versionado no Git.

Uso:
    uv run python scripts/gen_placeholder_model.py
    uv run python scripts/gen_placeholder_model.py --force  # sobrescreve mesmo se já existir

Por padrão, RECUSA sobrescrever um .joblib já existente em models/ — depois da OBS-001 Passo 5,
esse caminho pode conter o baseline real (ML-003), e rodar este script sem querer sobre ele
apagaria silenciosamente um artefato que levou minutos de treino real pra gerar de novo.
"""

import argparse
import logging
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TEXTS = [
    "patient stable minimal distress routine evaluation",
    "patient stable minimal distress routine checkup",
    "patient alert oriented moderate discomfort full workup",
    "patient alert oriented moderate discomfort planned",
    "patient critically ill immediate intervention required",
    "patient critically ill airway assessed vitals unstable",
]
LABELS = ["normal", "normal", "atencao", "atencao", "urgente", "urgente"]

OUTPUT_PATH = Path("models/tfidf_logreg_baseline.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="sobrescreve OUTPUT_PATH mesmo se já existir (ex: um baseline real gerado antes)",
    )
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        logger.info(
            "%s já existe — não sobrescrevendo (pode ser o baseline real da ML-003). "
            "Use --force se quiser mesmo gerar o placeholder no lugar dele.",
            OUTPUT_PATH,
        )
        return

    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))])
    pipeline.fit(TEXTS, LABELS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PATH)
    logger.info("Modelo placeholder salvo em %s (NÃO é o baseline real da ML-003)", OUTPUT_PATH)


if __name__ == "__main__":
    main()
