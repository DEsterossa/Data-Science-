"""Pydantic models for ask-history retrieval and debugging."""

from pydantic import BaseModel

from diarylens.ask_history.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DAY_HEADER_SKIP_CHARS,
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MIN_CHUNK_LEN,
    DEFAULT_MIN_DAY_INDEX_TEXT_LEN,
    DEFAULT_OVERLAP,
    DEFAULT_RAW_FALLBACK_CHARS,
    DEFAULT_TOP_K_CHUNKS,
    DEFAULT_TOP_K_DAYS,
    MODEL_NAME,
)


class AskHistoryError(Exception):
    """Raised when ask-history retrieval or generation cannot continue."""


class AskHistoryConfig(BaseModel):
    top_k_days: int = DEFAULT_TOP_K_DAYS
    top_k_chunks: int = DEFAULT_TOP_K_CHUNKS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP
    min_chunk_len: int = DEFAULT_MIN_CHUNK_LEN
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS
    day_header_skip_chars: int = DEFAULT_DAY_HEADER_SKIP_CHARS
    min_day_index_text_len: int = DEFAULT_MIN_DAY_INDEX_TEXT_LEN
    raw_fallback_chars: int = DEFAULT_RAW_FALLBACK_CHARS
    model_name: str = MODEL_NAME


class DayIndexRecord(BaseModel):
    date: str
    week_id: str | None = None
    source_day_md: str
    source_daily_json: str
    embedding_text: str
    embedding_text_len: int
    embedding_text_hash: str
    index_source: str


class DaySearchResult(BaseModel):
    rank: int
    date: str
    week_id: str | None = None
    score: float
    source_day_md: str
    source_daily_json: str
    embedding_text_preview: str | None = None


class EvidenceChunk(BaseModel):
    date: str
    week_id: str | None = None
    source_day_md: str
    chunk_index: int
    text: str


class EvidenceSearchResult(BaseModel):
    rank: int
    date: str
    week_id: str | None = None
    score: float
    source_day_md: str
    chunk_index: int
    text: str


class AskHistoryDebug(BaseModel):
    question: str
    config: AskHistoryConfig
    day_results: list[DaySearchResult]
    evidence_results: list[EvidenceSearchResult]
    evidence_context: str
    prompt: str
    answer: str | None = None
    answer_path: str | None = None
