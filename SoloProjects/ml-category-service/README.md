# ML Category Service (full-stack demo)

Учебный **production-like** демо-проект: модель классификации текста (TF-IDF + Logistic Regression) завернута в FastAPI и сопровождается веб-интерфейсом на React. Цель кейса для стажировки — показать путь от обученной модели до **demo-ready продукта**: API, UI, контейнеризация, конфигурация через переменные окружения.

## Что внутри

| Слой | Стек |
|------|------|
| Frontend | Vite, React 19, TypeScript, Tailwind CSS v4 |
| Backend | FastAPI, scikit-learn, joblib |
| Контейнеры | Docker, docker compose (backend + nginx со статикой) |

Пользователь вводит описание товара или объявления; сервис возвращает лучшую категорию, уверенность, задержку ответа и вероятности по всем классам. Интерфейс показывает метрики и распределение по категориям (прогресс-бары).

## Архитектура

```text
Браузер (React SPA)
       |  POST /predict, JSON
       v
FastAPI (инференс, измерение latency)
       |
       v
Артефакты: vectorizer, model, label_encoder (joblib)
```

Браузер обращается к API по адресу из `VITE_API_BASE_URL` (собирается на этапе **build** фронтенда). Backend разрешает запросы с указанных в `CORS_ORIGINS` origin (через запятую).

Границы проекта (намеренно простые): без Kubernetes, без отдельной БД и без auth — фокус на сквозном сценарии ML + API + UI.

## Структура репозитория

```text
ml-category-service/
  app/                 # FastAPI, схемы, загрузка модели
  models/              # joblib-артефакты (не в git по умолчанию, см. .gitignore)
  training/            # обучение
  tests/               # pytest для API
  frontend/            # SPA (Vite + React + TS)
  Dockerfile           # образ backend
  frontend/Dockerfile  # сборка статики + nginx
  docker-compose.yml
```

## Модель

Финальный пайплайн: **TF-IDF + LogisticRegression**.

- `max_features=10000`
- `ngram_range=(1, 1)`
- `stop_words="english"`

Ввод в UI и в поле `description` API ожидается **на английском языке** — так же обучалась модель и настроена векторизация.

Обоснование: баланс качества и скорости, быстрый inference, простая сериализация, предсказуемое поведение в учебном production-сценарии.

## Быстрый старт (Docker Compose)

Из корня репозитория (нужны собранные файлы в `models/` — см. обучение в `training/train.py`):

```bash
docker compose up --build --remove-orphans
```

Флаг `--remove-orphans` убирает старые контейнеры, если вы меняли имена сервисов в compose.

После запуска (порты по умолчанию):

- UI: [http://localhost:8080](http://localhost:8080)
- API: [http://localhost:8000](http://localhost:8000) (документация: `/docs`)
- Проверка здоровья: `GET /health`

Переменная `VITE_API_BASE_URL` для фронта задаётся **build-arg** в `docker-compose.yml` и совпадает с `BACKEND_HOST_PORT` из `.env` (по умолчанию `8000`). При смене порта бэкенда скопируйте [.env.example](.env.example) в `.env`, выставьте `BACKEND_HOST_PORT` и пересоберите фронт: `docker compose build --no-cache frontend`.

Backend в compose получает `CORS_ORIGINS`, включающий origin UI (`http://localhost:${FRONTEND_HOST_PORT}`).

### Если не запускается

1. **Ошибка `port is already allocated` (часто для 8000)** — другой процесс уже слушает порт. Варианты: остановить сервис на этом порту или скопировать `.env.example` в `.env` и задать, например, `BACKEND_HOST_PORT=8001`, затем `docker compose build --no-cache frontend` и снова `docker compose up`.
2. **Предупреждение про orphan containers** — выполните `docker compose up --remove-orphans` или `docker compose down --remove-orphans`.
3. **Backend падает при старте** — в каталоге `models/` должны лежать `vectorizer.joblib`, `model.joblib`, `label_encoder.joblib` (получите через `training/train.py`, если файлов ещё нет).

## Локальная разработка

**Backend:**

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Frontend** (в отдельном терминале):

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

По умолчанию Vite слушает порт 5173; убедитесь, что в `CORS_ORIGINS` backend есть `http://localhost:5173` (значение по умолчанию в коде это учитывает).

## Переменные окружения

Примеры см. в [.env.example](.env.example) и [frontend/.env.example](frontend/.env.example).

| Переменная | Где | Назначение |
|------------|-----|------------|
| `CORS_ORIGINS` | Backend | Список origin через запятую для браузерных запросов |
| `VITE_API_BASE_URL` | Frontend (build) | Базовый URL API без завершающего слэша |

## API

### `POST /predict`

**Request:**

```json
{
  "description": "Noise cancelling over-ear headphones with case"
}
```

**Response:**

```json
{
  "best_category": "Electronics",
  "best_confidence": 0.91,
  "latency_ms": 12.4,
  "predictions": [
    { "category": "Electronics", "confidence": 0.91 },
    { "category": "Household", "confidence": 0.05 },
    { "category": "Books", "confidence": 0.02 },
    { "category": "Clothing", "confidence": 0.02 }
  ]
}
```

## Тесты (backend)

```bash
pytest
```

Покрытие: `/health`, `/predict`, валидация пустого описания (422).

## О чём рассказать на собеседовании

- Обучение вынесено в `training/train.py`, артефакты версионируются как файлы joblib.
- Inference изолирован в `app/model.py`, API тонкий (`app/main.py`).
- Контракт запроса/ответа описан в Pydantic и продублирован типами TypeScript на фронте.
- Latency измеряется на уровне HTTP-обработчика; в UI отображаются уверенность и полное распределение по классам.
- CORS и базовый URL API вынесены в конфигурацию, а не зашиты в компоненты.
- Сборка фронта и два сервиса в `docker compose` дают воспроизводимое демо без ручной установки Node/Python на машине проверяющего.

## Возможные следующие шаги (не в scope текущего демо)

- Логирование запросов и структурированные ошибки
- Ограничение частоты запросов
- Прокси «один origin» (nginx перед API) для упрощения CORS в проде
