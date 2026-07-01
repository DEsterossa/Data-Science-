import json

import numpy as np

from diarylens.ask_history.embeddings import (
    build_or_load_day_embeddings,
    embedding_cache_key,
    load_embeddings_cache,
)
from diarylens.ask_history.models import DayIndexRecord, EvidenceChunk
from diarylens.ask_history.retrieval import search_days_embeddings, search_evidence_chunks


class FakeEmbeddingModel:
    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lower = text.lower()
            if "скука" in lower or "boredom" in lower:
                vector = np.array([1.0, 0.0], dtype=np.float32)
            elif "учеб" in lower or "study" in lower:
                vector = np.array([0.0, 1.0], dtype=np.float32)
            else:
                vector = np.array([0.5, 0.5], dtype=np.float32)
            vectors.append(vector)
        return np.vstack(vectors)


def _record(date: str, text: str, text_hash: str = "hash") -> DayIndexRecord:
    return DayIndexRecord(
        date=date,
        week_id="2026-W24",
        source_day_md=f"data/processed/days_md/{date}.md",
        source_daily_json=f"data/processed/days_json/{date}.json",
        embedding_text=text,
        embedding_text_len=len(text),
        embedding_text_hash=text_hash,
        index_source="daily_json",
    )


def test_search_days_embeddings_sorts_by_score_desc():
    records = [
        _record("2026-06-10", "учеба"),
        _record("2026-06-11", "скука"),
    ]
    embeddings = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    results = search_days_embeddings(
        "скука",
        records,
        embeddings,
        FakeEmbeddingModel(),
        top_k=2,
    )

    assert [result.date for result in results] == ["2026-06-11", "2026-06-10"]
    assert isinstance(results[0].score, float)
    assert "скука" in results[0].embedding_text_preview


def test_search_days_embeddings_respects_top_k():
    records = [_record("2026-06-10", "a"), _record("2026-06-11", "b")]
    embeddings = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    results = search_days_embeddings("скука", records, embeddings, FakeEmbeddingModel(), 1)

    assert len(results) == 1


def test_search_evidence_chunks_returns_ranked_results_with_cyrillic():
    chunks = [
        EvidenceChunk(
            date="2026-06-10",
            week_id="2026-W24",
            source_day_md="day1.md",
            chunk_index=0,
            text="Текст про учебу",
        ),
        EvidenceChunk(
            date="2026-06-11",
            week_id="2026-W24",
            source_day_md="day2.md",
            chunk_index=1,
            text="Текст про скуку",
        ),
    ]

    results = search_evidence_chunks("скука", chunks, FakeEmbeddingModel(), top_k=2)

    assert results[0].date == "2026-06-11"
    assert results[0].chunk_index == 1
    assert isinstance(results[0].score, float)
    assert "скуку" in results[0].text


def test_build_or_load_day_embeddings_writes_and_reads_cache(tmp_path):
    cache_path = tmp_path / "day_embeddings_cache.csv"
    record = _record("2026-06-11", "скука", text_hash="hash-1")
    model = FakeEmbeddingModel()

    first = build_or_load_day_embeddings([record], cache_path, "fake-model", model=model)
    second = build_or_load_day_embeddings(
        [record],
        cache_path,
        "fake-model",
        model=FakeEmbeddingModel(),
    )

    assert np.allclose(first, second)
    assert len(model.calls) == 1
    cache = load_embeddings_cache(cache_path)
    key = embedding_cache_key(
        record.date,
        record.source_day_md,
        record.source_daily_json,
        record.embedding_text_hash,
        "fake-model",
    )
    assert key in cache


def test_changed_hash_recomputes_only_stale_record(tmp_path):
    cache_path = tmp_path / "day_embeddings_cache.csv"
    first = _record("2026-06-11", "скука", text_hash="hash-1")
    build_or_load_day_embeddings([first], cache_path, "fake-model", model=FakeEmbeddingModel())
    changed = _record("2026-06-11", "учеба", text_hash="hash-2")
    model = FakeEmbeddingModel()

    embeddings = build_or_load_day_embeddings(
        [changed],
        cache_path,
        "fake-model",
        model=model,
    )

    assert len(model.calls) == 1
    assert np.allclose(embeddings[0], np.array([0.0, 1.0], dtype=np.float32))
    rows = cache_path.read_text(encoding="utf-8")
    assert "hash-2" in rows
    assert "hash-1" not in rows


def test_embedding_json_serializes_as_json_list(tmp_path):
    cache_path = tmp_path / "day_embeddings_cache.csv"
    record = _record("2026-06-11", "скука", text_hash="hash-1")

    build_or_load_day_embeddings([record], cache_path, "fake-model", model=FakeEmbeddingModel())

    content = cache_path.read_text(encoding="utf-8")
    assert json.loads(content.splitlines()[1].split(",", maxsplit=5)[5].rsplit(",", 1)[0])
