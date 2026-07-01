"""Semantic search over day records and raw evidence chunks."""

import numpy as np

from diarylens.ask_history.embeddings import encode_passages, encode_query
from diarylens.ask_history.models import (
    DayIndexRecord,
    DaySearchResult,
    EvidenceChunk,
    EvidenceSearchResult,
)


def _top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0 or top_k <= 0:
        return []
    limit = min(top_k, scores.size)
    return np.argsort(scores)[::-1][:limit].tolist()


def _preview(text: str, limit: int = 300) -> str:
    compact = " ".join(text.split())
    return compact[:limit]


def search_days_embeddings(
    query: str,
    records: list[DayIndexRecord],
    embeddings: np.ndarray,
    model,
    top_k: int,
) -> list[DaySearchResult]:
    """Search day embeddings and return the highest scoring days."""
    if not records or embeddings.size == 0:
        return []

    query_embedding = encode_query(model, query)
    scores = embeddings @ query_embedding

    results: list[DaySearchResult] = []
    for rank, index in enumerate(_top_indices(scores, top_k), start=1):
        record = records[index]
        results.append(
            DaySearchResult(
                rank=rank,
                date=record.date,
                week_id=record.week_id,
                score=float(scores[index]),
                source_day_md=record.source_day_md,
                source_daily_json=record.source_daily_json,
                embedding_text_preview=_preview(record.embedding_text),
            )
        )
    return results


def search_evidence_chunks(
    query: str,
    chunks: list[EvidenceChunk],
    model,
    top_k: int,
) -> list[EvidenceSearchResult]:
    """Search raw day chunks selected from the top day results."""
    if not chunks or top_k <= 0:
        return []

    chunk_embeddings = encode_passages(model, [chunk.text for chunk in chunks])
    query_embedding = encode_query(model, query)
    scores = chunk_embeddings @ query_embedding

    results: list[EvidenceSearchResult] = []
    for rank, index in enumerate(_top_indices(scores, top_k), start=1):
        chunk = chunks[index]
        results.append(
            EvidenceSearchResult(
                rank=rank,
                date=chunk.date,
                week_id=chunk.week_id,
                score=float(scores[index]),
                source_day_md=chunk.source_day_md,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
            )
        )
    return results
