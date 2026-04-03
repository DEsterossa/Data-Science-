import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_healthcheck():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_empty_description():
    payload = {
        "description": ""
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422