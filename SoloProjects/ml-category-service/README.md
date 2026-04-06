# ML Category Service

Fullstack demo: React frontend + FastAPI backend для классификации товара по `title`.

## Стек

- Frontend: Vite, React, TypeScript
- Backend: FastAPI, scikit-learn, joblib
- Model: TF-IDF + LogisticRegression
- Deploy: Docker / Docker Compose / Hugging Face Spaces (Docker)

## Контракт API

### `POST /predict`

Request:

```json
{
  "title": "wireless gaming mouse"
}
```

Response:

```json
{
  "best_category": "Electronics",
  "best_confidence": 0.91,
  "latency_ms": 12.4,
  "predictions": [
    { "category": "Electronics", "confidence": 0.91 },
    { "category": "Video Games", "confidence": 0.06 }
  ]
}
```

### Other endpoints

- `GET /health` -> `{"status":"ok"}`
- `GET /docs` -> Swagger UI

## Required model artifacts

The backend expects these files in `models/`:

- `vectorizer.joblib`
- `model.joblib`
- `label_encoder.joblib`

If they are missing, run training:

```bash
python training/train.py
```

## Local development

### Backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Docker Compose (local fullstack)

```bash
docker compose up --build
```

- UI: `http://localhost:8080`
- API: `http://localhost:8000`

## Hugging Face Spaces (Docker, fullstack in one container)

This repo now supports a single-container setup for Space:

- Nginx serves frontend static files
- Nginx proxies `/predict`, `/health`, `/docs`, `/openapi.json`, `/redoc` to FastAPI
- FastAPI runs via Uvicorn inside the same container

### Space repo

```bash
git clone https://huggingface.co/spaces/DEsterossa/ml-category-service
```

Copy project files into that repo (including `models/*.joblib`), commit, and push.

### Required files for Space

- `Dockerfile`
- `start-space.sh`
- `nginx-space.conf.template`
- `app/`
- `frontend/`
- `models/`
- `requirements.txt`

After build, open your Space URL and verify:

- `/` renders frontend
- `/health` returns status
- `/docs` is available
- `/predict` works with `title`
