---
title: Ml Category Service
emoji: 🚀
colorFrom: green
colorTo: pink
sdk: docker
pinned: false
---

# ML Category Service: Классификатор категорий товаров по `title`

### [➡️ Попробовать интерактивное демо на Hugging Face Spaces](https://huggingface.co/spaces/DEsterossa/ml-category-service)

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/DEsterossa/ml-category-service)

*(Примечание: при первичном заходе приложение может потребовать 1-2 минуты для "пробуждения" из спящего режима.)*

## Описание проекта
Агентная система предсказывает категорию товара по его `title`. Данные берутся из датасета e-commerce: модель обучается на текстовых заголовках и возвращает:

- `best_category` и `best_confidence`;
- `predictions` с top-5 категориями и вероятностями;
- `latency_ms` (время ответа сервиса).

## Навыки и инструменты
<img src="https://img.shields.io/badge/FastAPI-black?style=flat-square&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/React-black?style=flat-square&logo=react&logoColor=61DAFB" />
<img src="https://img.shields.io/badge/TypeScript-black?style=flat-square&logo=typescript&logoColor=blue" />
<img src="https://img.shields.io/badge/scikit-learn-black?style=flat-square&logo=scikit-learn&logoColor=F7931E" />
<img src="https://img.shields.io/badge/joblib-black?style=flat-square&logoColor=white" />
<img src="https://img.shields.io/badge/Docker-black?style=flat-square&logo=docker&logoColor=blue" />

## Сферы деятельности:
Обработка и анализ данных, Нейронные сети не используются (baseline на ML), Text Mining (NLP), Развертывание моделей (Deployment), MLflow эксперименты.

## Основные пункты исследования:
1. Подготовка данных: очистка, удаление пустых значений, разбиение train/test и label encoding.
2. Обучение baseline модели: `TfidfVectorizer` (n-grams) + `LogisticRegression` для multi-class классификации по `title`.
3. Оценка качества: accuracy, macro/weighted F1 и анализ распределения категорий.
4. Подготовка инференса: сервис возвращает top-k (top-5) категорий с вероятностями и измеряет `latency_ms`.
5. Развертывание: FastAPI backend + React frontend в одном Docker-контейнере, публикация на Hugging Face Spaces.

## Выводы и результаты
В рамках проекта собран учебный fullstack сервис, который демонстрирует полный путь от обучения текстового классификатора до интерактивного веб-приложения. Модель сохраняется артефактами:

- `models/vectorizer.joblib`
- `models/model.joblib`
- `models/label_encoder.joblib`

Сервис возвращает предсказания по входному `title`, включая `best_category`, `best_confidence`, список top-5 категорий и задержку ответа. Это подтверждает применимость простых, но эффективных ML-подходов для production-like demo на Hugging Face Spaces.
