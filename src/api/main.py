from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
import joblib
from prometheus_client import Counter, Histogram
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

# Métricas de negócio — OBS-001, Passo 2. Via prometheus_client direto (não pelo
# instrumentator, que é especializado em métricas HTTP genéricas). Funcionam como proxy
# de drift: não há rótulo verdadeiro disponível em produção, então a distribuição de
# classes previstas e a confiança do modelo são os sinais indiretos que temos.
PREDICTIONS_TOTAL = Counter(
    "triage_predictions_total",
    "Total de predições feitas pela API, por classe prevista.",
    ["classe_prevista"],
)

# RECALIBRADO na OBS-001 Passo 5 contra o baseline REAL (não mais hipótese): 90 chamadas
# reais a /predict, textos amostrados do dataset de verdade (30 por classe), confirmaram
# separação perfeita (100% de acerto, como a ML-003/ADR-002 já documentavam) — mas a
# confiança ficou muito mais colada em 1.0 do que a hipótese original previa: toda a
# massa caiu entre 0.9949 e 0.9997 (largura ~0.005!), não "alta confiança" em sentido
# genérico. Os buckets antigos (0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0) jogariam
# praticamente tudo nos 2 últimos buckets — mesmo problema de saturação já visto na
# calibração de latência do Passo 1, só que mais extremo aqui. Faixa baixa (0.34 a 0.9)
# mantida por segurança (nunca observada na prática, mas matematicamente possível se
# um texto não bater com nenhum padrão do template); resolução real concentrada em
# [0.99, 1.0], onde o baseline de fato vive.
PREDICTION_CONFIDENCE_BUCKETS = (
    0.34,
    0.5,
    0.7,
    0.9,
    0.95,
    0.98,
    0.99,
    0.993,
    0.995,
    0.996,
    0.997,
    0.998,
    0.9985,
    0.999,
    0.9993,
    0.9995,
    0.9997,
    0.9999,
    1.0,
)

PREDICTION_CONFIDENCE = Histogram(
    "triage_prediction_confidence",
    "Probabilidade (confiança) da classe prevista em cada chamada a /predict.",
    ["classe_prevista"],
    buckets=PREDICTION_CONFIDENCE_BUCKETS,
)


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

    PREDICTIONS_TOTAL.labels(classe_prevista=urgencia).inc()
    PREDICTION_CONFIDENCE.labels(classe_prevista=urgencia).observe(probabilidades[urgencia])

    return PredictResponse(urgencia=urgencia, probabilidades=probabilidades)
