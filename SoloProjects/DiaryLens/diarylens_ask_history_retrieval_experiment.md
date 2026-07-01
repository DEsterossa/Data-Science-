# DiaryLens: day-first retrieval experiment context

Дата фиксации: 2026-06-26  
Источник эксперимента: `experiments/day_first_retrieval.ipynb`  
Назначение файла: контекст для Cursor перед реализацией нового режима `ask_history` в основном проекте.

---

## 1. Короткий итог

В notebook был собран и проверен прототип нового режима DiaryLens:

```text
question
→ search relevant days through day_index
→ open raw days_md
→ local chunking inside selected days
→ search evidence chunks
→ build evidence_context
→ ask LLM
→ answer with analysis + quotes
```

Главный результат: **day-first + raw-first retrieval работает как основа для `ask_history`**.

Выбран основной retrieval-конфиг для MVP:

```python
DEFAULT_TOP_K_DAYS = 12
DEFAULT_TOP_K_CHUNKS = 12
DEFAULT_CHUNK_SIZE = 600
DEFAULT_OVERLAP = 100
DEFAULT_MAX_CHUNK_CHARS = 1200
```

Причина выбора: `k12_c600` даёт достаточно evidence для развёрнутого ответа с цитатами и покрытием нескольких дней. `k5_c600` лучше ранжирует верхушку, но часто даёт слишком мало материала для полноценного аналитического ответа.

---

## 2. Архитектурная рамка

Текущая стратегическая рамка проекта:

```text
day-first + raw-first retrieval
```

Роли артефактов:

```text
days_md = source of truth
days_json = day index / important events / topics / open loops
weekly_json = optional snapshot
weekly_report.md = optional readable snapshot
ask_history = main future direction
```

Важно для реализации:

- `days_md` остаётся источником истины.
- `days_json` используется как индекс дня, а не как полный пересказ.
- Retrieval должен находить двери к raw-дневникам, а не отвечать из summary JSON.
- Не нужно создавать новый LLM-summary JSON поверх daily/weekly JSON.
- Не нужно ломать текущий рабочий `run-week`.
- Новый режим `ask_history` должен быть отдельным слоем поверх уже созданных day-level artifacts.

Неправильное направление:

```text
daily JSON → weekly JSON → another summary JSON → answer
```

Правильное направление:

```text
days_json + days_md
→ day_index
→ semantic day search
→ raw day chunks
→ evidence-based answer
```

---

## 3. Данные и входные артефакты

В notebook использовались:

```text
data/processed/days_md/*.md
data/processed/days_json/*.json
```

Проверенный объём:

```text
days_md: 75 files
days_json: 75 files
all source_day_md links resolved: yes
```

Daily JSON содержит поля:

```text
date
week_id
source_day_md
important_moments
wins
tensions
emotions
body_energy_signals
study_signals
ml_ds_signals
social_signals
decisions
open_questions
key_quotes
short_summary
```

Покрытие daily JSON в целом достаточное для day index. При этом часть полей sparse:

- `important_moments`: 61/75 дней;
- `key_quotes`: 60/75 дней;
- `tensions`: 59/75 дней;
- `emotions`: 59/75 дней;
- `body_energy_signals`: 58/75 дней;
- `open_questions`: 19/75 дней;
- `ml_ds_signals`: 5/75 дней.

Вывод: daily JSON годится для первого retrieval-прототипа, но `open_questions` и `ml_ds_signals` в будущем можно улучшить через extraction prompt. Это не блокер для MVP.

---

## 4. Day index construction

Для каждого daily JSON собирается `embedding_text`.

Используемые поля:

```python
TEXT_FIELDS_FOR_DAY_INDEX = [
    "important_moments",
    "wins",
    "tensions",
    "emotions",
    "body_energy_signals",
    "study_signals",
    "ml_ds_signals",
    "social_signals",
    "decisions",
    "open_questions",
    "key_quotes",
]
```

В `embedding_text` добавляются:

- `date`;
- `week_id`;
- `short_summary`;
- списочные поля из `TEXT_FIELDS_FOR_DAY_INDEX`;
- для dict-item используются `quote` и `note`, если они есть.

Пример логики:

```text
date: 2026-06-12
week_id: 2026-W24
short_summary:
...
important_moments:
...
tensions:
...
key_quotes:
quote: ... | note: ...
```

### Fallback для провалившихся daily JSON

Часть daily JSON оказалась почти пустой, потому что LLM отказалась обрабатывать отдельные дни. Для таких дней добавлен raw fallback.

Параметры fallback:

```python
MIN_DAY_INDEX_TEXT_LEN = 200
RAW_FALLBACK_CHARS = 2500
```

Логика:

```text
if embedding_text_len < 200:
    embedding_text = date + week_id + first 2500 chars from raw days_md
```

Результат:

```text
normal daily JSON index: 61 days
raw markdown fallback: 14 days
```

Вывод: fallback важен, потому что поддерживает raw-first архитектуру. Даже если daily JSON сломан или пустой, день не выпадает из retrieval.

---

## 5. Модели и библиотеки

### TF-IDF baseline

Использовался как простой lexical baseline:

```python
TfidfVectorizer(
    lowercase=True,
    max_features=20_000,
    ngram_range=(1, 2),
)
```

Поиск:

```text
query → TF-IDF vector → cosine_similarity(query, day_texts)
```

### Semantic embeddings

Основная retrieval-модель:

```python
MODEL_NAME = "intfloat/multilingual-e5-small"
```

Причины выбора:

- поддерживает русский язык;
- подходит для retrieval;
- локально лёгкая модель;
- хороша как первый semantic baseline;
- не требует внешнего API для embeddings.

Для E5 обязательно использовать префиксы:

```python
"passage: " + day_text
"query: " + query
```

Embeddings нормализуются:

```python
normalize_embeddings=True
```

После нормализации cosine similarity считается через dot product:

```python
scores = embeddings @ query_embedding
```

В notebook использовался `sentence-transformers`; локально была проверена версия `5.3.0`.

### LLM для финального ответа

Использовался GigaChat через Python SDK:

```python
from gigachat import GigaChat
```

Переменные окружения:

```text
GIGACHAT_AUTH_KEY
GIGACHAT_MODEL_DAILY
```

В notebook использовался:

```python
GIGACHAT_VERIFY_SSL_CERTS = False
```

Это локальный workaround для проблем с сертификатами. Для production-режима лучше по возможности использовать нормальную проверку сертификатов.

---

## 6. Сравнение TF-IDF и embeddings

Мини-eval был сделан на двух запросах:

```text
скука
ds, машинное обучение, стажировка
```

Ручная шкала:

```text
2 = явно релевантно
1 = частично релевантно
0 = нерелевантно
```

Результаты:

| method | query | labels | weak_precision | strong_precision | avg_relevance | nDCG |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF | скука | [2, 0, 2, 0, 0] | 0.40 | 0.40 | 0.80 | 0.920 |
| TF-IDF | ds / ML / стажировка | [2, 1, 1, 1, 2] | 1.00 | 0.40 | 1.40 | 0.921 |
| embeddings | скука | [2, 2, 2, 0, 2] | 0.80 | 0.80 | 1.60 | 0.983 |
| embeddings | ds / ML / стажировка | [2, 2, 2, 1, 2] | 1.00 | 0.80 | 1.80 | 0.989 |

Вывод:

- TF-IDF иногда находит релевантные дни, но выдача заметно шумнее.
- Semantic embeddings лучше ловят смысловые совпадения.
- Embeddings лучше поднимают сильные совпадения наверх.
- Для `ask_history` основным day search должен быть semantic search, а TF-IDF можно оставить только как baseline/debug.

---

## 7. Two-stage retrieval

Итоговая retrieval-схема:

```text
Stage 1: query → semantic day search over day_index_df
Stage 2: top-k days → read raw days_md → local chunking → semantic chunk search
```

Почему не global chunk search сразу по всем дням:

- это обходит `days_json` как day index;
- нарушает day-first архитектуру;
- усложняет pipeline раньше времени;
- хуже объясняет, какие дни были найдены;
- превращает notebook в production RAG слишком рано.

Global chunk search можно рассмотреть позже как fallback/comparison, но не для текущего MVP.

---

## 8. Local chunking внутри найденных дней

Raw day markdown очищается:

- `\r\n` и `\r` заменяются на `\n`;
- удаляются HTML/markdown markers вида `<!-- page N -->`;
- лишние пустые строки схлопываются.

Используемая функция chunking:

```python
chunk_size = 600
overlap = 100
min_chunk_len = 350
```

Короткий хвостовой chunk мерджится в предыдущий chunk, если он короче `min_chunk_len`.

Причина: короткие chunks на 100–300 символов давали нестабильные semantic scores.

Также добавлен skip начала дня:

```python
DAY_HEADER_SKIP_CHARS = 130
```

Причина: в начале дневных markdown часто есть заголовки/метаданные. Иногда они содержали слова из запроса и давали header noise.

---

## 9. Evidence context

После поиска chunks формируется `evidence_context` для LLM.

Каждый найденный chunk содержит:

```text
rank
date
week_id
score
source_day_md
chunk_index
text
```

`max_chunk_chars` в контексте ограничивает длину каждого chunk перед отправкой в LLM.

Для MVP выбран:

```python
DEFAULT_MAX_CHUNK_CHARS = 1200
```

Это нужно, чтобы:

- LLM видела достаточно сырого контекста;
- chunks не раздували prompt бесконечно;
- ответ мог включать цитаты из raw markdown.

---

## 10. Prompt для ask_history

Вывод LLM был настроен через prompt. После нескольких итераций выбран стиль:

```text
вдумчивый и честный собеседник
не сухой RAG-report
анализ дневниковых фрагментов
цитаты вплетены в основной ответ
большой блок цитат/evidence в конце
```

Ключевые требования к prompt:

- ответ должен быть приятным для чтения;
- не просто пересказ retrieved chunks;
- не ставить диагнозы;
- не выдумывать факты вне найденных фрагментов;
- если данных мало, говорить, что вывод предварительный;
- цитаты нужно вплетать в основной анализ;
- в конце должен быть отдельный блок `Цитаты и evidence по дням`;
- финальный evidence-блок должен содержать минимум 7 цитат или близких фрагментов, если данных хватает;
- если найденных фрагментов меньше 7, использовать все доступные и прямо написать, что evidence меньше 7 пунктов.

Структура ответа:

```text
# Ответ
## Суть найденных фрагментов
## Что здесь происходит эмоционально и психологически
## На что стоит обратить внимание
## Возможные искажения или самообман
## Практический вывод
## Один короткий вопрос для самоанализа
## Цитаты и evidence по дням
```

Важный вывод по prompt: **не надо приклеивать evidence кодом после LLM-ответа**. Это формально добавляет цитаты, но не показывает, что сам ответ был построен на них. Правильнее давать chunks в prompt и требовать, чтобы LLM естественно вплетала короткие цитаты в основной анализ, а затем выводила большой evidence-блок в конце.

---

## 11. Retrieval parameter experiment

Цель: подобрать параметры retrieval, при которых `ask_history` получает хорошие evidence chunks.

Проверялись запросы:

```python
retrieval_test_queries = [
    "Когда я чувствую скуку?",
    "Какие у меня есть интересы?",
    "Когда я обычно болею?",
    "Когда я чувствую радость?",
]
```

Проверялись конфиги:

```python
retrieval_configs = [
    {"config_id": "k5_c600", "top_k_days": 5, "top_k_chunks": 5, "chunk_size": 600, "overlap": 100},
    {"config_id": "k7_c800", "top_k_days": 7, "top_k_chunks": 7, "chunk_size": 800, "overlap": 150},
    {"config_id": "k7_c500", "top_k_days": 7, "top_k_chunks": 7, "chunk_size": 500, "overlap": 100},
    {"config_id": "k12_c600", "top_k_days": 12, "top_k_chunks": 12, "chunk_size": 600, "overlap": 100},
]
```

Разметка chunks:

```text
2 = сильный evidence для запроса
1 = частично полезный / связанный контекст
0 = шум или слабая связь
```

Количество размеченных строк: 124.

---

## 12. Метрики retrieval

Оценивалось качество именно retrieval, а не качество LLM-ответа.

Использованные метрики:

### `weak_precision_at_k`

Доля chunks с `label >= 1`.

Показывает, сколько найденных chunks хотя бы полезны.

### `strong_precision_at_k`

Доля chunks с `label == 2`.

Показывает, сколько найденных chunks являются сильным evidence.

### `avg_relevance_at_k`

Средний label.

Интуитивная средняя полезность выдачи:

```text
0.0 = мусор
1.0 = частично полезно
2.0 = почти идеально
```

### `nDCG_at_k`

Главная метрика ранжирования.

Показывает, стоят ли сильные chunks наверху. Значение ближе к `1.0` означает лучшее ранжирование.

Важный нюанс: `nDCG@5`, `nDCG@7` и `nDCG@12` сравнивать напрямую можно только осторожно. Top-12 сложнее держать чистым, чем top-5.

### `unique_dates_at_k`

Сколько разных дней попало в найденные chunks.

Для DiaryLens это важно, потому что:

- 1 день = локальный эпизод;
- 2 дня = повторяющийся сигнал;
- 3+ дня = можно осторожно говорить о теме среди найденных дней.

### `avg_text_len`

Средняя длина найденных chunks.

Помогает понять, не слишком ли chunks короткие или широкие.

---

## 13. Результаты retrieval experiment

Итоговая таблица по конфигам:

| config_id | top_k_days | top_k_chunks | chunk_size | overlap | mean_nDCG | mean_strong_precision | mean_weak_precision | mean_avg_relevance | mean_unique_dates | mean_text_len | queries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k5_c600 | 5 | 5 | 600 | 100 | 0.963109 | 0.450000 | 0.800000 | 1.250000 | 3.50 | 598.65 | 4 |
| k12_c600 | 12 | 12 | 600 | 100 | 0.909461 | 0.458333 | 0.875000 | 1.333333 | 8.00 | 627.94 | 4 |
| k7_c500 | 7 | 7 | 500 | 100 | 0.853490 | 0.500000 | 0.857143 | 1.357143 | 4.75 | 564.50 | 4 |
| k7_c800 | 7 | 7 | 800 | 150 | 0.822783 | 0.464286 | 0.821429 | 1.285714 | 5.25 | 756.68 | 4 |

---

## 14. Интерпретация результатов

### `k5_c600`

```python
top_k_days = 5
top_k_chunks = 5
chunk_size = 600
overlap = 100
```

Лучший по качеству ранжирования:

```text
mean_nDCG = 0.963109
```

Сильные chunks чаще стоят наверху. Это хороший compact/debug mode.

Минус: только 5 chunks. Для нового формата ответа с развёрнутым анализом, встроенными цитатами и 7+ цитатами в конце этого часто мало.

### `k12_c600`

```python
top_k_days = 12
top_k_chunks = 12
chunk_size = 600
overlap = 100
```

Лучший баланс для полноценного `ask_history`:

```text
mean_weak_precision = 0.875000
mean_avg_relevance = 1.333333
mean_unique_dates = 8.00
```

Он чуть хуже ранжирует верхушку, чем `k5_c600`, но даёт больше полезного материала и покрывает больше разных дней.

Это важно для DiaryLens, потому что большинство вопросов не просто про один точный факт, а про состояние, повторяемость, паттерны и периодические темы.

### `k7_c500`

```python
top_k_days = 7
top_k_chunks = 7
chunk_size = 500
overlap = 100
```

Самый плотный по сильным evidence:

```text
mean_strong_precision = 0.500000
mean_avg_relevance = 1.357143
```

Chunks по 500 символов дают больше конкретики и меньше лишнего контекста. Но ранжирование слабее:

```text
mean_nDCG = 0.853490
```

Можно держать как precision mode.

### `k7_c800`

```python
top_k_days = 7
top_k_chunks = 7
chunk_size = 800
overlap = 150
```

Показал самый слабый `nDCG` среди проверенных вариантов:

```text
mean_nDCG = 0.822783
```

Вероятно, chunks по 800 символов начинают смешивать несколько мыслей. Для текущего MVP не выбран.

---

## 15. Финальное решение по default config

Для MVP выбран:

```python
DEFAULT_TOP_K_DAYS = 12
DEFAULT_TOP_K_CHUNKS = 12
DEFAULT_CHUNK_SIZE = 600
DEFAULT_OVERLAP = 100
DEFAULT_MAX_CHUNK_CHARS = 1200
```

Причина:

- лучше подходит для развёрнутого анализа;
- даёт больше evidence для цитат;
- лучше покрывает разные дни;
- сохраняет приемлемое качество ранжирования;
- соответствует продуктовой задаче `ask_history`, где важен не только один точный chunk, но и несколько подтверждающих дневниковых фрагментов.

Дополнительные режимы, которые можно оставить в коде как presets:

```python
COMPACT_RETRIEVAL_CONFIG = {
    "top_k_days": 5,
    "top_k_chunks": 5,
    "chunk_size": 600,
    "overlap": 100,
}

PRECISION_RETRIEVAL_CONFIG = {
    "top_k_days": 7,
    "top_k_chunks": 7,
    "chunk_size": 500,
    "overlap": 100,
}

FULL_ASK_HISTORY_CONFIG = {
    "top_k_days": 12,
    "top_k_chunks": 12,
    "chunk_size": 600,
    "overlap": 100,
    "max_chunk_chars": 1200,
}
```

---

## 16. Что реализовать в Python

Реализация должна быть маленьким отдельным режимом, не переписыванием MVP.

Минимальная первая версия может быть в одном файле:

```text
src/diarylens/ask_history.py
```

Позже можно разнести на модули:

```text
src/diarylens/ask_history/
  __init__.py
  day_index.py
  embeddings.py
  retrieval.py
  prompt.py
  gigachat_client.py
  pipeline.py
```

Но для первого переноса из notebook лучше не распиливать слишком рано.

### Главный pipeline

```text
ask_history(question)
→ load/build day_index_df
→ load/build day_embeddings
→ semantic day search
→ read top-k raw days_md
→ local chunking
→ semantic evidence chunk search
→ build evidence_context
→ build prompt
→ call GigaChat
→ return answer
```

### Рекомендуемый API

Минимально:

```python
def ask_history(question: str) -> str:
    ...
```

Лучше сразу предусмотреть debug:

```python
def ask_history(
    question: str,
    return_debug: bool = False,
    config: AskHistoryConfig | None = None,
) -> str | dict:
    ...
```

При `return_debug=True` вернуть:

```python
{
    "question": question,
    "answer": answer,
    "evidence_results": evidence_results,
    "evidence_context": evidence_context,
    "prompt": prompt,
    "config": config,
}
```

Debug-режим важен, потому что пользователь должен видеть, какие chunks попали в ответ.

---

## 17. Кэширование embeddings

Не нужно пересчитывать day embeddings при каждом вопросе.

Рекомендуемые файлы:

```text
data/index/day_index.csv или data/index/day_index.parquet
data/index/day_embeddings.npy
```

Минимальная логика:

```text
if index files exist and are fresh:
    load day_index + embeddings
else:
    rebuild from days_json + days_md fallback
```

Для MVP можно сделать простую команду или функцию rebuild без сложного freshness tracking.

В будущем можно добавить проверку по file mtimes или hash, но сейчас это не обязательно.

---

## 18. CLI-режим

После реализации функции можно добавить CLI:

```bash
diarylens ask-history "Когда я чувствую скуку?"
```

или:

```bash
diarylens ask-history --question "Когда я чувствую скуку?"
```

Полезные CLI flags:

```text
--top-k-days
--top-k-chunks
--chunk-size
--overlap
--debug
```

Но для первого шага достаточно дефолтов и `--debug`.

---

## 19. Ограничения эксперимента

Важно не переоценивать результаты.

Ограничения:

- ручная разметка субъективна;
- тестовых query всего 4;
- оценивался retrieval, а не финальное качество LLM-ответа;
- LLM answer eval пока не проведён;
- `nDCG@5`, `nDCG@7`, `nDCG@12` сравниваются осторожно, потому что top-k разный;
- embeddings-модель выбрана как practical baseline, не как финальная лучшая модель;
- chunking по символам прост, но уже работает достаточно хорошо для MVP.

Что не нужно делать сейчас:

- Qdrant;
- FAISS;
- vector database;
- web app;
- автоматический judge;
- полноценный eval framework;
- новый summary JSON;
- переписывание weekly pipeline.

---

## 20. Cursor implementation constraints

При реализации в Cursor важно соблюдать ограничения:

- не ломать `diarylens run-week`;
- не менять существующие daily/weekly schemas без отдельной причины;
- использовать существующие `days_md` и `days_json`;
- `days_md` считать source of truth;
- `days_json` считать day index;
- не добавлять новые технологии без запроса;
- не превращать notebook experiment в сложный production RAG;
- сначала перенести рабочую notebook-логику как MVP-модуль;
- после реализации перечислить изменённые файлы и результаты тестов.

---

## 21. Suggested next implementation step

Первый безопасный шаг:

```text
перенести функции из notebook в src/diarylens/ask_history.py
```

Минимальный набор функций:

```python
resolve_source_day_md
load_daily_records
build_day_embedding_text
build_day_index_df
read_day_raw_text
build_fallback_embedding_text
load_embedding_model
build_or_load_day_embeddings
search_days_embeddings
normalize_text
chunk_text_by_chars
build_local_chunks_from_days
search_evidence_chunks
build_evidence_context
build_ask_history_prompt
call_gigachat
ask_history
```

После этого добавить простой CLI wrapper.

---

## 22. Final status

Notebook-прототип можно считать завершённым на уровне:

```text
day-first retrieval prototype completed
```

Следующая стадия:

```text
build Python MVP ask_history mode
```

Цель Python MVP:

```text
question → evidence-backed diary answer from raw days_md
```
