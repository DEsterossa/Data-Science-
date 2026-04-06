import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "load_artifacts", lambda: None)

    def fake_predict(_title: str):
        return {
            "best_category": "Electronics",
            "best_confidence": 0.91,
            "predictions": [
                {"category": "Electronics", "confidence": 0.91},
                {"category": "Video Games", "confidence": 0.06},
            ],
        }

    monkeypatch.setattr(main_module, "predict", fake_predict)

    with TestClient(app) as test_client:
        yield test_client


def test_healthcheck(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_success_with_title(client):
    payload = {"title": "wireless gaming mouse"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert body["best_category"] == "Electronics"
    assert isinstance(body["best_confidence"], float)
    assert isinstance(body["latency_ms"], float)
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) >= 1
    assert set(body["predictions"][0].keys()) == {"category", "confidence"}


def test_predict_empty_title_validation_error(client):
    payload = {"title": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_with_description_only_is_invalid(client):
    payload = {"description": "wireless gaming mouse"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422