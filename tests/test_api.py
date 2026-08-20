from pathlib import Path

from fastapi.testclient import TestClient
import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.api.main import DEFAULT_MODEL_PATH, app

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
