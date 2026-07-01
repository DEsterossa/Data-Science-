"""Embedding model loading and CSV cache helpers."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from diarylens.ask_history.models import AskHistoryError, DayIndexRecord

EMBEDDINGS_CACHE_COLUMNS = [
    "date",
    "source_day_md",
    "source_daily_json",
    "embedding_text_hash",
    "model_name",
    "embedding_json",
    "created_at",
]


def load_embedding_model(model_name: str):
    """Load the sentence-transformers model used by ask-history."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise AskHistoryError(
            "sentence-transformers is required for ask-history. "
            "Install project dependencies or run `pip install -e .`."
        ) from exc

    return SentenceTransformer(model_name)


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return array.astype(np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).astype(np.float32)


def _encode(model, texts: list[str]) -> np.ndarray:
    try:
        encoded = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except TypeError:
        encoded = model.encode(texts)

    array = np.asarray(encoded, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return _normalize_rows(array)


def encode_passages(model, texts: list[str]) -> np.ndarray:
    """Encode E5 passages with normalized vectors."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return _encode(model, [f"passage: {text}" for text in texts])


def encode_query(model, query: str) -> np.ndarray:
    """Encode an E5 query with a normalized vector."""
    return _encode(model, [f"query: {query}"])[0]


def embedding_cache_key(
    date: str,
    source_day_md: str,
    source_daily_json: str,
    embedding_text_hash: str,
    model_name: str,
) -> str:
    """Build a freshness-aware cache key for one day embedding."""
    return (
        f"{date}|{source_day_md}|{source_daily_json}|"
        f"{embedding_text_hash}|{model_name}"
    )


def _record_cache_key(record: DayIndexRecord, model_name: str) -> str:
    return embedding_cache_key(
        record.date,
        record.source_day_md,
        record.source_daily_json,
        record.embedding_text_hash,
        model_name,
    )


def _row_cache_key(row: dict) -> str:
    return embedding_cache_key(
        row["date"],
        row["source_day_md"],
        row.get("source_daily_json", ""),
        row["embedding_text_hash"],
        row["model_name"],
    )


def _load_embeddings_cache_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not row.get("embedding_json"):
                continue
            rows[_row_cache_key(row)] = row
    return rows


def load_embeddings_cache(path: Path) -> dict[str, np.ndarray]:
    """Load embedding vectors from a CSV cache keyed by day/hash/model."""
    rows = _load_embeddings_cache_rows(path)
    cache: dict[str, np.ndarray] = {}
    for key, row in rows.items():
        cache[key] = np.array(json.loads(row["embedding_json"]), dtype=np.float32)
    return cache


def save_embeddings_cache(path: Path, cache_rows: list[dict]) -> None:
    """Persist embedding cache rows as UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EMBEDDINGS_CACHE_COLUMNS)
        writer.writeheader()
        for row in cache_rows:
            serialized = dict(row)
            vector = serialized.get("embedding_json")
            if not isinstance(vector, str):
                serialized["embedding_json"] = json.dumps(
                    np.asarray(vector, dtype=np.float32).tolist(),
                    ensure_ascii=False,
                )
            writer.writerow(serialized)


def _cache_row_for_record(
    record: DayIndexRecord,
    model_name: str,
    embedding: np.ndarray,
    created_at: str | None = None,
) -> dict:
    return {
        "date": record.date,
        "source_day_md": record.source_day_md,
        "source_daily_json": record.source_daily_json,
        "embedding_text_hash": record.embedding_text_hash,
        "model_name": model_name,
        "embedding_json": json.dumps(embedding.tolist(), ensure_ascii=False),
        "created_at": created_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def build_or_load_day_embeddings(
    records: list[DayIndexRecord],
    cache_path: Path,
    model_name: str,
    model=None,
) -> np.ndarray:
    """Return one embedding per day record, using CSV cache when fresh."""
    if not records:
        return np.empty((0, 0), dtype=np.float32)

    cache_rows = _load_embeddings_cache_rows(cache_path)
    vectors: list[np.ndarray | None] = [None] * len(records)
    output_rows: list[dict | None] = [None] * len(records)
    missing_indices: list[int] = []

    for index, record in enumerate(records):
        key = _record_cache_key(record, model_name)
        cached_row = cache_rows.get(key)
        if cached_row is None:
            missing_indices.append(index)
            continue

        vectors[index] = np.array(
            json.loads(cached_row["embedding_json"]),
            dtype=np.float32,
        )
        output_rows[index] = cached_row

    if missing_indices:
        active_model = model or load_embedding_model(model_name)
        texts = [records[index].embedding_text for index in missing_indices]
        encoded = encode_passages(active_model, texts)
        for offset, record_index in enumerate(missing_indices):
            vector = encoded[offset]
            record = records[record_index]
            vectors[record_index] = vector
            output_rows[record_index] = _cache_row_for_record(
                record,
                model_name,
                vector,
            )

    final_vectors = [vector for vector in vectors if vector is not None]
    final_rows = [row for row in output_rows if row is not None]
    save_embeddings_cache(cache_path, final_rows)
    return np.vstack(final_vectors).astype(np.float32)
