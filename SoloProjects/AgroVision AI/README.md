# AgroVision AI: Классификатор заболеваний растений

### [➡️ Попробовать интерактивное демо на Hugging Face Spaces](https://huggingface.co/spaces/DEsterossa/agrovision_ai)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/DEsterossa/agrovision-ai)

*(Примечание: При первичном заходе приложению может потребоваться 1-2 минуты для "пробуждения" из спящего режима.)*

## Описание проекта
Агрохолдинг «Зелёный Урожай» разрабатывает систему компьютерного зрения для помощи фермерам и агрономам. Внедрение автоматической диагностики по фотографиям листьев позволит:

- На ранней стадии определять заболевания растений, чтобы оперативно принять меры и спасти урожай;
- Снизить затраты на экспертов-агрономов и автоматизировать процесс мониторинга полей;
- Применять химикаты более точечно, уменьшая их общее использование и повышая экологичность продукции.

## Навыки и инструменты
<img src="https://img.shields.io/badge/PyTorch-black?style=flat-square&logo=pytorch&logoColor=orange"/><img src="https://img.shields.io/badge/Pandas-black?style=flat-square&logo=pandas&logoColor=orange"/><img src="https://img.shields.io/badge/NumPy-black?style=flat-square&logo=numpy&logoColor=blue"/><img src="https://img.shields.io/badge/Matplotlib-black?style=flat-square&logo=matplotlib&logoColor=white"/><img src="https://img.shields.io/badge/Flask-black?style=flat-square&logo=flask&logoColor=white"/><img src="https://img.shields.io/badge/Docker-black?style=flat-square&logo=docker&logoColor=blue"/>

## Сферы деятельности:
Обработка и анализ данных, Нейронные сети, Computer Vision, Развертывание моделей (Deployment)

## Основные пункты исследования:
1.  Подготовка данных и настройка конвейера аугментации.
2.  Обучение модели методом Transfer Learning (Feature Extraction).
3.  Тонкая настройка модели (Fine-Tuning) для достижения максимальной точности.
4.  Анализ результатов, визуализация предсказаний и тестирование.
5.  Создание и развертывание интерактивного веб-приложения на Flask и Hugging Face Spaces.

## Выводы и результаты
В рамках проекта была разработана и обучена модель на базе архитектуры **ResNet50** с использованием двухэтапного трансферного обучения. После тонкой настройки (fine-tuning) модель продемонстрировала выдающуюся способность к классификации 38 различных видов заболеваний и состояний растений, достигнув **точности 99%** на предоставленном тестовом наборе данных.

Для демонстрации работы модели было создано интерактивное веб-приложение на Flask, которое развернуто на платформе Hugging Face Spaces. Данный результат подтверждает возможность интеграции модели в реальные агротехнические решения.
