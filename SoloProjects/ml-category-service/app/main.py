import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model import load_artifacts, predict
from app.schemas import PredictionRequest, PredictionResponse


def _parse_cors_origins() -> list[str]:
    # Comma-separated list, e.g. "http://localhost:5173,http://localhost:8080"
    default = (
        "http://localhost:5173,"
        "http://localhost:4173,"
        "http://localhost:8080"
    )
    raw = os.getenv("CORS_ORIGINS", default)
    return [part.strip() for part in raw.split(",") if part.strip()]


app = FastAPI(title='ML Category service')

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.on_event('startup')
def startup_event():
    load_artifacts()

@app.get('/health')
def healthcheck():
    return {'status': 'ok'}

@app.post('/predict', response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    start_time = time.perf_counter()

    result = predict(request.description)

    latency_ms = (time.perf_counter() - start_time) * 1000
    result['latency_ms'] = round(latency_ms, 2)

    return PredictionResponse(**result)