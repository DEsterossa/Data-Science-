"""Configuration constants for ask-history retrieval."""

DEFAULT_TOP_K_DAYS = 12
DEFAULT_TOP_K_CHUNKS = 12
DEFAULT_CHUNK_SIZE = 600
DEFAULT_OVERLAP = 100
DEFAULT_MIN_CHUNK_LEN = 350
DEFAULT_MAX_CHUNK_CHARS = 1200
DEFAULT_DAY_HEADER_SKIP_CHARS = 130
DEFAULT_MIN_DAY_INDEX_TEXT_LEN = 200
DEFAULT_RAW_FALLBACK_CHARS = 2500

MODEL_NAME = "intfloat/multilingual-e5-small"

COMPACT_RETRIEVAL_CONFIG = {
    "top_k_days": 5,
    "top_k_chunks": 5,
    "chunk_size": 600,
    "overlap": 100,
    "max_chunk_chars": 1200,
}

PRECISION_RETRIEVAL_CONFIG = {
    "top_k_days": 7,
    "top_k_chunks": 7,
    "chunk_size": 500,
    "overlap": 100,
    "max_chunk_chars": 1200,
}

FULL_ASK_HISTORY_CONFIG = {
    "top_k_days": DEFAULT_TOP_K_DAYS,
    "top_k_chunks": DEFAULT_TOP_K_CHUNKS,
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "overlap": DEFAULT_OVERLAP,
    "max_chunk_chars": DEFAULT_MAX_CHUNK_CHARS,
}

RETRIEVAL_PRESET_VALUES = {
    "compact": COMPACT_RETRIEVAL_CONFIG,
    "precision": PRECISION_RETRIEVAL_CONFIG,
    "full": FULL_ASK_HISTORY_CONFIG,
}

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
