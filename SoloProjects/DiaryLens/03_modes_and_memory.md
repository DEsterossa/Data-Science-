# DiaryLens — режимы работы и память

## 1. Новая рамка памяти

Главная единица памяти — **день**, а не неделя.

```text
day → period view / week view → trends / retrieval answers
```

Неделя остаётся удобной группировкой, но не должна быть главным источником истины.

Правильная рамка:

```text
daily markdown = source of truth
daily JSON = day index
weekly JSON = optional snapshot
retrieval index = search layer
```

---

## 2. Текущий режим v0.1

Текущий рабочий режим кода:

### `weekly_full`

```text
один PDF недели
→ full pipeline
→ weekly_report.md
```

Использовать как стабилизированный MVP и способ получать текущие day-level артефакты.

Но продуктово `weekly_full` больше не должен считаться единственной или главной функцией.

---

## 3. Главный будущий режим

### `ask_history`

Главный режим развития проекта.

```text
question
→ retrieve relevant days / raw chunks
→ answer with evidence
```

Примеры:

```text
Когда у меня чаще всего появлялась скука?
Как менялось моё отношение к DiaryLens?
Какие open loops тянутся дольше всего?
Какие дни похожи на текущий период?
Что обычно происходило перед распылением?
```

Главное правило:

> LLM должна отвечать по найденным raw day fragments, а не только по summaries.

---

## 4. Другие будущие режимы

Эти режимы — roadmap, а не текущая реализация.

### `search_memory`

Технический режим для проверки retrieval:

```text
query → top-k days / chunks
```

Вывод:

- score;
- date;
- source_day_md;
- text preview;
- doc_id.

---

### `similar_days`

Поиск похожих дней:

```text
day embedding / query embedding
→ top-k similar days
```

Полезно для поиска повторяющихся состояний, событий и сценариев.

---

### `similar_periods`

Поиск похожих периодов:

```text
несколько дней / текущая неделя / вопрос
→ похожие прошлые дни или группы дней
```

Важно: похожий период лучше собирать из найденных дней, а не только из weekly summaries.

---

### `trend_check`

Проверка динамики по теме:

```text
тема / вопрос
→ найденные дни по времени
→ краткая динамика
```

Примеры:

```text
как менялась тема ML?
как часто появлялась скука?
как развивался DiaryLens?
```

---

### `open_loops`

Отдельный режим по незакрытым вопросам:

```text
найти open questions / decisions в daily JSON
→ проверить raw days
→ показать, что тянется долго
```

---

### `weekly_snapshot`

Короткий недельный обзор.

```text
7 дней → readable weekly snapshot
```

Это полезный output, но не главный продукт.

---

## 5. Иерархия памяти

## 1. Day

День — основная ячейка памяти.

День нужен для:

- точности;
- фактов;
- evidence;
- локальных событий;
- поиска похожих дней;
- восстановления исходного контекста;
- ответов на вопросы через raw text.

В текущем v0.1 day markdown уже существует в:

```text
data/processed/days_md/YYYY-MM-DD.md
```

Daily JSON уже существует в:

```text
data/processed/days_json/YYYY-MM-DD.json
```

Daily JSON должен восприниматься как day index:

```text
important events
topics
open questions
key quotes
short summary
```

Не дублировать здесь полную Pydantic-схему.

Актуальная схема определяется кодом:

```text
src/diarylens/schemas.py
```

---

## 2. Week

Неделя — не главная ячейка памяти, а удобный view над днями.

Неделя нужна для:

- weekly snapshot;
- группировки дней;
- приблизительной навигации по времени;
- быстрых human-readable обзоров.

В текущем v0.1 weekly JSON уже существует в:

```text
data/processed/weeks_json/{week_id}.json
```

Но weekly JSON не должен быть главным источником для retrieval answers.

Если вопрос требует деталей, система должна открыть relevant day markdown.

---

## 3. Period

Период — гибкая группировка дней.

Период может быть:

- неделей;
- несколькими днями;
- месяцем;
- произвольным диапазоном;
- набором похожих дней, найденных retrieval.

Period-level summaries можно добавлять позже, но они не должны заменять raw day evidence.

---

## 4. Month

Месяц — будущий уровень обзора.

Месяц нужен не для деталей, а для сжатой исторической картины:

- recurring patterns;
- repeated tensions;
- progress by area;
- important decisions;
- unresolved questions;
- changes from previous month.

Monthly aggregation **не является частью v0.1**.

---

## 5. Long-term patterns

`long_term_patterns.md` — curated memory: сжатые паттерны, подтверждённые пользователем.

Пример:

```markdown
- В периоды карьерной неопределённости часто появляется желание срочно добавить новое направление.
  Confidence: medium
  Evidence: 2025-03, 2025-09, 2026-02
  Status: accepted
```

Эта память должна обновляться осторожно.

Не после каждого единичного случая, а только если паттерн повторился несколько раз.

Пользователь должен подтверждать изменения:

```text
accept / reject / edit
```

Long-term memory **не является частью MVP v0.1**.

---

## 6. Retrieval memory

Retrieval index — это не источник истины.

Источник истины:

```text
data/processed/days_md/
```

Навигационный слой:

```text
data/processed/days_json/
```

Retrieval layer:

```text
data/memory/day_index.jsonl
data/memory/chunks.jsonl
data/memory/embeddings.npy
data/memory/metadata.jsonl
```

Retrieval index может хранить:

- embeddings;
- metadata;
- короткий payload;
- путь к полному day markdown;
- путь к daily JSON.

На первом этапе не нужен Qdrant / vector DB.

Достаточно локального индекса и cosine similarity.

---

## 7. Что embedding-ить

Не нужно embedding-ить весь JSON как есть.

Не нужно создавать ещё один LLM-summary JSON.

Лучше иметь два типа embedding documents.

## 1. Day index embedding

Собирается из daily JSON:

```text
date
short_summary
important_events
topics
open_questions
key_quotes
```

Используется, чтобы найти релевантные дни.

---

## 2. Raw chunk embedding

Собирается из day markdown chunks:

```text
date
week_id
raw text chunk
optional lightweight tags from daily JSON
```

Используется, чтобы найти конкретные фрагменты для ответа.

Главное правило:

> embedding_text может помогать найти источник, но не должен заменять источник.

---

## 8. Защита от data leakage

Если в системе лежат 2 года дневника, модель не должна использовать будущее при анализе старого периода.

Это решается metadata-фильтрами.

Для каждого объекта должны быть даты:

```text
date
start_date
end_date
created_at
source_period
```

Если пользователь спрашивает про период до определённой даты, retrieval должен брать только допустимые источники.

Пример:

```text
анализируем март 2025
→ retrieval берёт только дни до конца марта 2025, если вопрос требует исторически честного анализа
```

Если пользователь явно спрашивает “как это развивалось позже”, тогда можно использовать будущие дни.

---

## 9. Roadmap памяти

### v0.1 — Day-Level Artifacts + Optional Weekly Snapshot

```text
current weekly PDF
→ days_md
→ daily JSON
→ optional weekly JSON
→ optional weekly_report.md
```

Фокус:

- стабильный local pipeline;
- нормальный split into days;
- качественный daily JSON as day index;
- сохранение raw day markdown.

---

### v0.2 — Day-First Retrieval Prototype

```text
days_md + days_json
→ day_index
→ raw chunks
→ embeddings
→ search_memory
→ ask_history
```

Фокус:

- локальный индекс без Qdrant;
- cosine similarity;
- ответы по raw chunks;
- evidence с датами и цитатами;
- ручная оценка retrieval quality.

---

### v0.3 — Historical Backfill + Searchable Diary Memory

```text
2 года PDF
→ days
→ daily JSON index
→ raw chunk index
→ embeddings
→ searchable memory
```

На этом этапе можно просто построить память.

Не обязательно сразу делать сложные режимы.

---

### v0.4 — Retrieval Modes and Pattern Analysis

```text
ask_history
+ similar_days
+ trend_check
+ open_loops
+ similar_periods
```

Добавить режимы:

- `ask_history`;
- `similar_days`;
- `similar_periods`;
- `open_loops`;
- `trend_check`;
- `weekly_snapshot`.

---

## 10. Что не предлагать как следующий шаг

Пока не предлагать без явного запроса:

- Telegram bot;
- web app;
- fine-tuning;
- long-term memory;
- monthly aggregation;
- PDF/DOCX export;
- сложные dashboards;
- сложный multi-service RAG;
- Qdrant / vector DB как обязательный первый шаг;
- ещё один summary JSON поверх daily/weekly JSON.

Текущий фокус:

```text
stabilize day-level artifacts
→ prototype day-first retrieval
→ ask_history with evidence
```
