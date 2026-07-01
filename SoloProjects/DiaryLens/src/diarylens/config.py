"""Paths and project layout for DiaryLens."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_WEEKLY_DIR = DATA_DIR / "raw" / "weekly"
RAW_ARCHIVE_DIR = DATA_DIR / "raw" / "archive"
INTERIM_RAW_TEXT_DIR = DATA_DIR / "interim" / "raw_text"
INTERIM_CLEAN_TEXT_DIR = DATA_DIR / "interim" / "clean_text"
DAYS_MD_DIR = DATA_DIR / "processed" / "days_md"
DAY_MANIFESTS_DIR = DATA_DIR / "processed" / "day_manifests"
DAYS_JSON_DIR = DATA_DIR / "processed" / "days_json"
DAYS_JSON_FAILED_DIR = DATA_DIR / "processed" / "days_json_failed"
WEEKS_JSON_DIR = DATA_DIR / "processed" / "weeks_json"
REPORTS_DIR = DATA_DIR / "reports"
DATA_CONTEXT_DIR = DATA_DIR / "context"
MEMORY_DIR = DATA_DIR / "memory"
ANSWERS_DIR = DATA_DIR / "answers"
ASK_HISTORY_ANSWERS_DIR = ANSWERS_DIR / "ask_history"
GOALS_CONTEXT_PATH = DATA_CONTEXT_DIR / "goals.md"
USER_CONTEXT_PATH = DATA_CONTEXT_DIR / "context_about_me.md"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

PROJECT_DIRS = (
    RAW_WEEKLY_DIR,
    RAW_ARCHIVE_DIR,
    INTERIM_RAW_TEXT_DIR,
    INTERIM_CLEAN_TEXT_DIR,
    DAYS_MD_DIR,
    DAY_MANIFESTS_DIR,
    DAYS_JSON_DIR,
    DAYS_JSON_FAILED_DIR,
    WEEKS_JSON_DIR,
    REPORTS_DIR,
    DATA_CONTEXT_DIR,
    MEMORY_DIR,
    ASK_HISTORY_ANSWERS_DIR,
    PROMPTS_DIR,
)


def ensure_project_dirs() -> None:
    """Create project data directories if they do not exist."""
    for path in PROJECT_DIRS:
        path.mkdir(parents=True, exist_ok=True)
