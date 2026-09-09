from pathlib import Path

from fastapi.testclient import TestClient
import joblib
from prometheus_client import REGISTRY
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.api.main import DEFAULT_MODEL_PATH, app

CLASSES = ("normal", "atencao", "urgente")

pytestmark = pytest.mark.api

SYNTHETIC_TEXTS = [
    "patient stable minimal distress routine evaluation",
    "patient stable minimal distress routine checkup",
    "patient alert oriented moderate discomfort full workup",
    "patient alert oriented moderate discomfort planned",
    "patient critically ill immediate intervention required",
    "patient critically ill airway assessed vitals unstable",
]
SYNTHETIC_LABELS = ["normal", "normal", "atencao", "atencao", "urgente", "urgente"]


@pytest.fixture
def synthetic_model_path(tmp_path: Path) -> Path:
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))])
    pipeline.fit(SYNTHETIC_TEXTS, SYNTHETIC_LABELS)
    model_path = tmp_path / "synthetic_baseline.joblib"
    joblib.dump(pipeline, model_path)
    return model_path


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, synthetic_model_path: Path) -> TestClient:
    monkeypatch.setenv("MODEL_PATH", str(synthetic_model_path))
    with TestClient(app) as test_client:
        yield test_client


def test_predict_returns_valid_class_and_probabilities(client: TestClient) -> None:
    response = client.post("/predict", json={"clinical_notes": "patient stable minimal distress"})
    assert response.status_code == 200
    body = response.json()
    assert body["urgencia"] in {"normal", "atencao", "urgente"}
    assert set(body["probabilidades"]) == {"normal", "atencao", "urgente"}
    assert sum(body["probabilidades"].values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_urgencia_matches_highest_probability(client: TestClient) -> None:
    response = client.post("/predict", json={"clinical_notes": "patient critically ill"})
    body = response.json()
    assert body["urgencia"] == max(body["probabilidades"], key=body["probabilidades"].get)


def test_predict_rejects_blank_text(client: TestClient) -> None:
    response = client.post("/predict", json={"clinical_notes": "   "})
    assert response.status_code == 422


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_exposes_red_metrics_after_a_request(client: TestClient) -> None:
    client.post("/predict", json={"clinical_notes": "patient stable minimal distress"})

    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert 'http_requests_total{handler="/predict",method="POST",status="2xx"}' in body
    assert "http_request_duration_seconds_bucket" in body
    assert "http_request_duration_highr_seconds_bucket" in body


def test_metrics_excludes_health_checks(client: TestClient) -> None:
    client.get("/health")

    body = client.get("/metrics").text

    # /health é chamado a cada 30s pelo HEALTHCHECK do Docker (DOCK-001) — se aparecesse
    # aqui, inflaria a métrica de "taxa de requisições" com heartbeat de infraestrutura
    # em vez de tráfego de negócio (OBS-001, ver spec, seção Riscos).
    assert '"/health"' not in body


def test_metrics_exposes_business_metrics_after_a_request(client: TestClient) -> None:
    client.post("/predict", json={"clinical_notes": "patient critically ill"})

    body = client.get("/metrics").text

    assert "triage_predictions_total" in body
    assert "triage_prediction_confidence_bucket" in body


def test_predict_increments_business_metrics_for_the_predicted_class(
    client: TestClient,
) -> None:
    # Captura o "antes" das 3 classes possíveis — só depois da chamada sabemos qual classe
    # o modelo previu, então não dá para assumir de antemão qual contador vai mudar.
    before_counts = {
        classe: REGISTRY.get_sample_value("triage_predictions_total", {"classe_prevista": classe})
        or 0.0
        for classe in CLASSES
    }
    before_confidence_sums = {
        classe: REGISTRY.get_sample_value(
            "triage_prediction_confidence_sum", {"classe_prevista": classe}
        )
        or 0.0
        for classe in CLASSES
    }

    response = client.post("/predict", json={"clinical_notes": "patient critically ill"})
    body = response.json()
    urgencia = body["urgencia"]
    confianca = body["probabilidades"][urgencia]

    count_depois = REGISTRY.get_sample_value(
        "triage_predictions_total", {"classe_prevista": urgencia}
    )
    assert count_depois == before_counts[urgencia] + 1

    for outra_classe in CLASSES:
        if outra_classe != urgencia:
            # Uma série com label só nasce no primeiro .labels(...) observado — se a classe
            # nunca foi prevista nesta sessão de testes, get_sample_value devolve None, não
            # 0.0. Mesma normalização usada para capturar o "antes".
            count_outra = (
                REGISTRY.get_sample_value(
                    "triage_predictions_total", {"classe_prevista": outra_classe}
                )
                or 0.0
            )
            assert count_outra == before_counts[outra_classe]

    # O histograma registrou exatamente a probabilidade que a API devolveu no corpo da
    # resposta — não só "algum número", a confiança de verdade daquela predição.
    soma_confianca_depois = REGISTRY.get_sample_value(
        "triage_prediction_confidence_sum", {"classe_prevista": urgencia}
    )
    assert soma_confianca_depois == pytest.approx(before_confidence_sums[urgencia] + confianca)


def test_startup_fails_with_clear_message_when_model_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    with pytest.raises(FileNotFoundError, match="notebooks/02_baseline.ipynb"), TestClient(app):
        pass


@pytest.mark.slow
def test_predict_with_real_baseline_model() -> None:
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip(
            f"modelo real não encontrado em {DEFAULT_MODEL_PATH} — "
            "rode notebooks/02_baseline.ipynb antes deste teste"
        )
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"clinical_notes": "67yo M c/o Chest pain. Patient in moderate distress."},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["urgencia"] in {"normal", "atencao", "urgente"}
    assert set(body["probabilidades"]) == {"normal", "atencao", "urgente"}
