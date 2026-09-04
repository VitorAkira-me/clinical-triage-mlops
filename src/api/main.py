from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, Request
import joblib

from src.api.schemas import HealthResponse, PredictRequest, PredictResponse

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "tfidf_logreg_baseline.joblib"
)


def get_model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


def load_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. Rode notebooks/02_baseline.ipynb "
            "(ML-003) para gerar o artefato, ou defina a variável de ambiente MODEL_PATH "
            "apontando para um .joblib já treinado."
        )
    return joblib.load(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.pipeline = load_model(get_model_path())
    yield


app = FastAPI(title="Clinical Triage API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, req: Request) -> PredictResponse:
    pipeline = req.app.state.pipeline
    proba = pipeline.predict_proba([request.clinical_notes])[0]
    probabilidades = {
        str(classe): float(p) for classe, p in zip(pipeline.classes_, proba, strict=True)
    }
    urgencia = max(probabilidades, key=probabilidades.get)
    return PredictResponse(urgencia=urgencia, probabilidades=probabilidades)
