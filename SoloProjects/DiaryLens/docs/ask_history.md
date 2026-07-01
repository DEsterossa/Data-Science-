# Ask History

## Purpose

`ask_history` позволяет задавать вопросы к прошлому дневнику и получать ответ по найденным raw day markdown фрагментам.

## Architecture

```text
question
-> day index search
-> raw day chunk search
-> LLM answer with evidence
```

Первый поиск работает по `days_json` как навигационному day index. Второй поиск открывает исходные `days_md`, режет только выбранные дни на chunks и ищет evidence chunks.

## Source of truth

`data/processed/days_md/*.md` - source of truth. Daily JSON используется как индекс важных событий, тем и open loops, но не заменяет raw дневник.

## Index files

```text
data/memory/day_index.csv
data/memory/day_embeddings_cache.csv
data/memory/ask_history_answers/*.md
```

`day_index.csv` хранит текст для поиска по дням, source paths, hash и источник индекса (`daily_json` или `raw_fallback`).

`day_embeddings_cache.csv` хранит embedding cache для day index. Если hash текста изменился, embedding пересчитывается только для изменившегося дня.

`ask_history_answers/*.md` хранит markdown-ответы, полученные через `diarylens ask-history`.

## CLI

```bash
diarylens ask-history "Когда я чувствую скуку?"
diarylens ask-history --question "Когда я чувствую скуку?" --debug
diarylens search-memory "Когда я чувствую скуку?"
```

Полезные флаги:

```text
--force-rebuild-index
--top-k-days
--top-k-chunks
--chunk-size
--overlap
--preset compact|precision|full
```

## Debugging

`search-memory` не вызывает LLM. Команда строит или загружает day index, загружает cached day embeddings, ищет релевантные дни, строит chunks из raw markdown и печатает top evidence chunks.

`ask-history --debug` печатает найденные дни, найденные chunks и затем markdown answer.

## Limitations

- Retrieval quality depends on embeddings.
- No Qdrant, FAISS or vector database yet.
- Chunk embeddings are computed on the fly from top-k days.
- Answer quality depends on retrieved evidence.
- No automatic eval yet.
- GigaChat still receives selected evidence chunks, so sensitive diary fragments can leave the local machine during answer generation.
