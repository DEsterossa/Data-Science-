"""Clean raw diary markdown into normalized clean markdown."""

from pathlib import Path

from diarylens.config import INTERIM_CLEAN_TEXT_DIR, INTERIM_RAW_TEXT_DIR


class TextCleaningError(Exception):
    """Raised when text cleaning fails."""


def _validate_week_id(week_id: str | None) -> str:
    if not week_id or not week_id.strip():
        raise TextCleaningError("week_id is required")
    return week_id.strip()


def raw_text_input_path(week_id: str, input_dir: Path | None = None) -> Path:
    base = input_dir or INTERIM_RAW_TEXT_DIR
    return base / f"{week_id}_raw.md"


def clean_text_output_path(week_id: str, output_dir: Path | None = None) -> Path:
    base = output_dir or INTERIM_CLEAN_TEXT_DIR
    return base / f"{week_id}_clean.md"


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _update_header_line(line: str, week_id: str) -> str:
    if line.startswith("# Raw diary text:"):
        return f"# Clean diary text: {week_id}"
    return line


def clean_text_content(raw_content: str, week_id: str) -> str:
    """Apply minimal cleaning rules to raw markdown content."""
    if not raw_content.strip():
        raise TextCleaningError("Raw markdown is empty")

    normalized = _normalize_line_endings(raw_content)
    lines = [_update_header_line(line.rstrip(), week_id) for line in normalized.split("\n")]

    collapsed: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                collapsed.append("")
            continue

        blank_run = 0
        collapsed.append(line)

    while collapsed and collapsed[0] == "":
        collapsed.pop(0)
    while collapsed and collapsed[-1] == "":
        collapsed.pop()

    return "\n".join(collapsed) + "\n"


def clean_raw_markdown(
    week_id: str,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Read raw markdown for a week and save cleaned markdown."""
    week_id = _validate_week_id(week_id)
    raw_path = raw_text_input_path(week_id, input_dir)

    if not raw_path.exists():
        raise TextCleaningError(f"Raw markdown not found: {raw_path}")

    raw_content = raw_path.read_text(encoding="utf-8")
    clean_content = clean_text_content(raw_content, week_id)

    out_dir = output_dir or INTERIM_CLEAN_TEXT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = clean_text_output_path(week_id, out_dir)
    output_path.write_text(clean_content, encoding="utf-8")
    return output_path
