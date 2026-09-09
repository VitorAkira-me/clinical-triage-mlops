from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
import joblib
from prometheus_fastapi_instrumentator import Instrumentator, metrics

from src.api.schemas import HealthResponse, PredictRequest, PredictResponse

# Buckets de latência (segundos) calibrados para um modelo linear leve (TF-IDF +
# LogisticRegression): a predição isolada mede ~0.4ms, então os buckets padrão da lib
# (o menor é 0.01s = 10ms) jogariam quase toda requisição no mesmo bucket, sem
# resolução para calcular p95 de verdade (OBS-001, RNF5). Usados nos dois histogramas
# de `metrics.default()`: o "highr" (sem label de rota, mais preciso) e o "lowr" (tem
# o label `handler`, é o que permite p95 por rota — por padrão vem com só 3 buckets
# grosseiros, 0.1/0.5/1s, inúteis pra latência na casa de milissegundos).
LATENCY_BUCKETS_SECONDS = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1,
    2.5,
    5,
    10,
)

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

# Métricas RED (Rate, Errors, Duration) automáticas — OBS-001, Passo 1.
# `/health` e `/metrics` ficam fora: o HEALTHCHECK do Docker bate em /health a cada 30s
# indefinidamente e inflaria a métrica de "taxa de requisições" com heartbeat de
# infraestrutura, não tráfego de negócio (ver docs/specs/OBS-001.md, seção Riscos).
instrumentator = Instrumentator(excluded_handlers=["/health", "/metrics"])
instrumentator.add(
    metrics.default(
        latency_highr_buckets=LATENCY_BUCKETS_SECONDS,
        latency_lowr_buckets=LATENCY_BUCKETS_SECONDS,
    )
)
instrumentator.instrument(app).expose(app, endpoint="/metrics")


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
