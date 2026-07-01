# DiaryLens / Journal Analyzer — суть проекта

## Главная формулировка

DiaryLens / Journal Analyzer — это локальная система структурирования и поиска по личному дневнику.

Система получает дневниковые PDF, извлекает из них текст, разбивает записи на дни, сохраняет исходный дневной markdown как **source of truth**, строит лёгкий daily JSON с важными событиями / темами / open loops и позволяет задавать вопросы к прошлому через retrieval:

```text
diary PDF
→ raw markdown
→ clean markdown
→ days md
→ daily JSON as day index
→ retrieval over relevant days / chunks
→ LLM answer with evidence
```

Это не “ИИ, который меня понимает”.

Это **аналитическая поисковая система по личному дневнику**.

Она должна помогать видеть:

- что реально происходило;
- какие темы и события повторялись;
- какие дни / периоды похожи друг на друга;
- где был прогресс;
- где пользователь буксовал;
- какие open loops остались;
- как менялись темы, состояния, проекты и решения во времени;
- что можно осторожно заключить на основе evidence.

---

# 1. Главный принцип

Система должна работать по принципу:

```text
raw diary → day-level structure → retrieval → answer with evidence
```

Не так:

```text
raw diary → daily JSON → weekly JSON → another summary JSON → answer
```

Потому что каждый дополнительный слой пересказа сжимает дневник и теряет детали.

Правильная рамка:

```text
PDF / дневниковые записи
→ raw markdown
→ clean markdown
→ days md
→ daily JSON with important events / topics / open questions
→ embeddings / search over days and raw chunks
→ answer from original day markdown fragments
```

То есть:

- `days_md` — главный источник истины;
- `daily JSON` — навигационный слой: события, темы, сигналы, open loops;
- `weekly JSON` — опциональная недельная сводка, но не главный источник правды;
- retrieval должен находить релевантные дни / фрагменты, а ответ должен строиться по raw text.

---

# 2. Что это за система

Journal Analyzer — это система анализа и поиска по личному дневнику.

Её задача — не заменять психолога и не давать мотивационные советы, а помогать пользователю трезво видеть:

- что происходило в конкретные дни;
- какие события были важными;
- какие темы повторялись;
- какие вопросы тянутся долго;
- как менялись проекты, фокус, энергия и настроение;
- какие периоды похожи друг на друга;
- какие выводы можно сделать только на основе найденных evidence.

Обычный текущий вход — один PDF с дневником за неделю.

Но стратегически неделя больше не считается главной аналитической единицей. Неделя — это удобная группировка дней. Главная единица памяти — **день**.

---

# 3. Главная функция продукта

Главная функция DiaryLens должна постепенно смещаться от weekly report к `ask_history`:

```text
вопрос пользователя
→ поиск релевантных дней / фрагментов / событий
→ открытие исходного day markdown
→ LLM-ответ с датами, цитатами и ограничениями
```

Примеры вопросов:

```text
Когда у меня чаще всего появлялась скука?
Как менялось моё отношение к DiaryLens?
Какие события обычно предшествовали распылению?
Какие open loops тянутся дольше всего?
Когда спорт реально помогал состоянию?
На какие прошлые периоды похожа эта неделя?
```

Финальный ответ должен быть:

```text
вывод → evidence → осторожная интерпретация → ограничения
```

Не “я знаю, что с тобой”.

А:

```text
по найденным записям видно вот это; данных может быть недостаточно вот здесь
```

---

# 4. Личная и портфельная цель

Проект нужен одновременно в двух ролях.

## 1. Личный инструмент

DiaryLens должен помогать:

- задавать вопросы к прошлым дневниковым данным;
- находить релевантные дни и периоды;
- видеть повторяющиеся темы;
- не терять open loops;
- отличать реальные факты от интерпретаций;
- проверять выводы через evidence;
- видеть динамику по времени, а не только одну неделю.

## 2. ML/LLM engineering проект для портфолио

Проект должен демонстрировать:

- обработку PDF и неструктурированного текста;
- построение промежуточных артефактов;
- day-level structured extraction;
- prompt chaining там, где он оправдан;
- JSON / Pydantic validation;
- raw-first evidence-based answering;
- embeddings / semantic search;
- retrieval over personal time-series data;
- evaluation retrieval quality;
- local-first file-based pipeline;
- работу с CursorAI как инструментом разработки.

Важно позиционировать проект не как “чат-бот для дневника” и не как “генератор weekly review”, а как **local-first LLM/NLP pipeline for searchable diary memory**.

---

# 5. Главная мысль

DiaryLens должен быть не “эмоциональным анализатором”, а системой поиска и аналитики личной истории:

> он помогает находить факты, повторяющиеся темы, похожие периоды, open loops и осторожные выводы на основе исходных дневниковых записей.

Система не должна анализировать ради анализа.

Её цель — помогать пользователю отличать:

- реальные факты от интерпретаций;
- исходный текст от сжатого пересказа;
- важное от второстепенного;
- повторяющийся паттерн от единичного эпизода;
- evidence-based вывод от красивой догадки.

---

# 6. Текущий статус

```text
MVP v0.1 / stabilization stage
```

Core pipeline уже работает end-to-end:

```text
PDF
→ raw markdown
→ clean markdown
→ days md
→ daily JSON
→ verified daily JSON
→ weekly JSON
→ weekly_report.md
```

Главная команда текущего кода:

```bash
diarylens run-week --week-id 2026-W22 --pdf data/raw/weekly/2026-W22.pdf
```

Текущий output:

```text
data/reports/{week_id}_weekly_report.md
```

Но продуктовая стратегия меняется:

```text
weekly_report.md = optional weekly snapshot
ask_history / retrieval over days = main future direction
```

---

# 7. Практический фокус разработки

Разработка ведётся через CursorAI.

Задачи для Cursor должны быть маленькими, техническими и проверяемыми.

Новая приоритетная рамка:

```text
не переписывать весь MVP
не добавлять ещё один summary JSON
сначала построить day-first retrieval prototype
```

Ближайший технический фокус:

```text
days_md + days_json
→ day index / chunks
→ local embeddings
→ search relevant days
→ answer from raw day markdown with evidence
```

---

# 8. Roadmap

## v0.1 — PDF to Day-Level Artifacts + Weekly Snapshot

Цель:

```text
один PDF недели
→ extract text
→ split into days
→ daily markdown
→ daily JSON
→ optional weekly JSON
→ optional weekly_report.md
```

Статус: core pipeline работает, идёт stabilization.

---

## v0.2 — Day-First Retrieval Prototype

Цель:

```text
days_md + days_json
→ retrieval documents / chunks
→ embeddings
→ search-memory
→ ask-history answer with evidence
```

Фокус:

- день как главная единица памяти;
- raw day markdown как source of truth;
- daily JSON как набор важных событий, тем и open loops;
- локальный простой индекс без Qdrant;
- ответы по найденным raw fragments, а не по очередному summary.

---

## v0.3 — Historical Backfill + Day Memory Index

Цель:

```text
2 года дневника
→ days
→ day JSON index
→ raw chunk index
→ embeddings
→ searchable diary memory
```

Исторический архив будет загружаться несколькими PDF-файлами с неопределённой длительностью по дням.

---

## v0.4 — Retrieval Modes and Pattern Analysis

Цель:

```text
question
→ time-aware retrieval from day chunks / day events
→ answer with evidence
```

Возможные режимы:

- `ask_history`;
- `similar_days`;
- `similar_periods`;
- `open_loops`;
- `trend_check`;
- `weekly_snapshot`.

---

# 9. Что пока не делать

Пока не нужно без явного запроса:

- делать Telegram-бота;
- делать web app;
- делать fine-tuning;
- строить сложный RAG / Qdrant / vector DB;
- добавлять long-term memory;
- делать monthly aggregation;
- делать PDF/DOCX export;
- строить сложные dashboards;
- пытаться анализировать все 2 года до стабилизации day-level pipeline;
- хранить весь дневник в одном промпте;
- делать ещё один LLM-summary JSON поверх daily/weekly JSON;
- превращать проект в психологического ассистента.

Сначала нужно добиться, чтобы `ask_history` находил релевантные дни и отвечал по raw text лучше, чем ручной поиск по дневнику.
