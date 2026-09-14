"""AIR-001 — DAG mínima: carrega o dataset fedmml-ed-triage e treina o baseline
TF-IDF + LogisticRegression (mesma lógica da ML-003 / notebooks/02_baseline.ipynb).

Sob demanda (schedule=None) — o requisito oficial não pede retreino agendado, só uma DAG
funcional que rode de ponta a ponta com sucesso. Ver docs/specs/AIR-001.md.
"""

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path

from airflow.decorators import dag, task

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/opt/airflow/project")
RAW_PARQUET = PROJECT_ROOT / "data" / "raw" / "fedmml_ed_triage_raw.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "tfidf_logreg_baseline.joblib"

DATASET_ID = "olaflaitinen/fedmml-ed-triage"
DATASET_FILENAME = "fedmml_ed_triage_dataset.csv"

CLASSES = ["normal", "atencao", "urgente"]
SEED = 42


def _remap_esi(esi: int) -> str:
    if esi in (1, 2):
        return "urgente"
    if esi == 3:
        return "atencao"
    if esi in (4, 5):
        return "normal"
    raise ValueError(f"ESI fora do intervalo esperado (1-5): {esi!r}")


@dag(
    dag_id="air001_train_baseline",
    description="AIR-001 — carrega fedmml-ed-triage e treina o baseline TF-IDF + LogReg (ML-003)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["air-001", "ml-003"],
)
def air001_train_baseline() -> None:
    @task
    def load_dataset() -> str:
        """Resolve o CSV do dataset com o menor esforço possível: parquet local já
        existente > cache do Hugging Face (sem rede) > download real com HF_TOKEN.
        """
        import pandas as pd

        if RAW_PARQUET.exists():
            logger.info("%s já existe — reusando, sem baixar de novo.", RAW_PARQUET)
            return str(RAW_PARQUET)

        from huggingface_hub import hf_hub_download

        try:
            csv_path = hf_hub_download(
                repo_id=DATASET_ID,
                filename=DATASET_FILENAME,
                repo_type="dataset",
                local_files_only=True,
            )
            logger.info("CSV resolvido do cache local do Hugging Face (sem rede).")
        except Exception:
            logger.info("Sem cache local do dataset — baixando com HF_TOKEN do .env.")
            from dotenv import load_dotenv
            from huggingface_hub import login

            load_dotenv(PROJECT_ROOT / ".env")
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError(
                    "Dataset não está em cache local e HF_TOKEN não encontrado em .env — "
                    "sem como baixar o fedmml-ed-triage (repositório gated)."
                )
            login(token=token)
            csv_path = hf_hub_download(
                repo_id=DATASET_ID, filename=DATASET_FILENAME, repo_type="dataset"
            )

        df = pd.read_csv(csv_path)
        RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(RAW_PARQUET, index=False)
        logger.info("Salvo %s (%d linhas).", RAW_PARQUET, len(df))
        return str(RAW_PARQUET)

    @task
    def train_baseline(raw_parquet_path: str) -> str:
        """Mesma lógica de treino da ML-003: split 80/20 estratificado (seed=42),
        TF-IDF + LogisticRegression(class_weight="balanced"), salva o .joblib.
        """
        import joblib
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline

        df = pd.read_parquet(raw_parquet_path)
        df["urgencia"] = df["esi_level"].map(_remap_esi)
        df = df.dropna(subset=["clinical_notes"]).reset_index(drop=True)

        train_df, _ = train_test_split(
            df, test_size=0.2, stratify=df["urgencia"], random_state=SEED
        )

        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                (
                    "clf",
                    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
                ),
            ]
        )
        pipeline.fit(train_df["clinical_notes"], train_df["urgencia"])

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        logger.info("Modelo salvo em %s.", MODEL_PATH)
        return str(MODEL_PATH)

    train_baseline(load_dataset())


air001_train_baseline()
