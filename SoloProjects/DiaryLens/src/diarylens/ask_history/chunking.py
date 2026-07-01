"""Raw day markdown normalization and local chunking."""

import re
from pathlib import Path

from diarylens import config as project_config
from diarylens.ask_history.models import (
    AskHistoryConfig,
    AskHistoryError,
    DaySearchResult,
    EvidenceChunk,
)

PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+\d+\s*-->", flags=re.IGNORECASE)
BLANK_LINES_RE = re.compile(r"\n{3,}")


def _resolve_existing_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path

    project_candidate = project_config.PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    return project_candidate


def normalize_raw_day_text(text: str) -> str:
    """Normalize raw day markdown without changing its content semantics."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = PAGE_MARKER_RE.sub("", normalized)
    normalized = BLANK_LINES_RE.sub("\n\n", normalized)
    return normalized.strip()


def chunk_text_by_chars(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chunk_len: int,
) -> list[str]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    stripped = text.strip()
    if not stripped:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(stripped)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = stripped[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap

    if len(chunks) > 1 and len(chunks[-1]) < min_chunk_len:
        chunks[-2] = f"{chunks[-2]}\n{chunks[-1]}".strip()
        chunks.pop()

    return chunks


def build_local_chunks_from_days(
    day_results: list[DaySearchResult],
    config: AskHistoryConfig,
) -> list[EvidenceChunk]:
    """Read selected raw days and split them into searchable chunks."""
    chunks: list[EvidenceChunk] = []
    for day_result in day_results:
        source_day_md = _resolve_existing_path(day_result.source_day_md)
        if not source_day_md.exists():
            raise AskHistoryError(
                f"source day markdown not found: {day_result.source_day_md}"
            )

        text = source_day_md.read_text(encoding="utf-8")
        text = normalize_raw_day_text(text)
        text = text[config.day_header_skip_chars :]
        day_chunks = chunk_text_by_chars(
            text,
            config.chunk_size,
            config.overlap,
            config.min_chunk_len,
        )
        for chunk_index, chunk_text in enumerate(day_chunks):
            chunks.append(
                EvidenceChunk(
                    date=day_result.date,
                    week_id=day_result.week_id,
                    source_day_md=day_result.source_day_md,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )
    return chunks
