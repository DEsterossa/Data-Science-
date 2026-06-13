"""Extract text from weekly PDF into raw markdown."""

from pathlib import Path

from pypdf import PdfReader

from diarylens.config import INTERIM_RAW_TEXT_DIR, PROJECT_ROOT


class PdfExtractionError(Exception):
    """Raised when PDF text extraction fails."""


def _validate_week_id(week_id: str | None) -> str:
    if not week_id or not week_id.strip():
        raise PdfExtractionError("week_id is required")
    return week_id.strip()


def _validate_pdf_path(pdf_path: Path) -> Path:
    resolved = pdf_path.resolve()
    if not resolved.exists():
        raise PdfExtractionError(f"PDF file not found: {pdf_path}")
    if resolved.suffix.lower() != ".pdf":
        raise PdfExtractionError(f"Not a PDF file: {pdf_path}")
    return resolved


def _source_pdf_display_path(pdf_path: Path) -> str:
    try:
        relative = pdf_path.resolve().relative_to(PROJECT_ROOT.resolve())
        return relative.as_posix()
    except ValueError:
        return pdf_path.as_posix()


def _extract_pages_text(pdf_path: Path) -> list[str]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise PdfExtractionError(f"Failed to open PDF: {pdf_path}") from exc

    if not reader.pages:
        raise PdfExtractionError(f"No pages in PDF: {pdf_path}")

    pages_text: list[str] = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")

    if not any(text.strip() for text in pages_text):
        raise PdfExtractionError(f"No text extracted from PDF: {pdf_path}")

    return pages_text


def _format_raw_markdown(
    week_id: str,
    pdf_path: Path,
    pages_text: list[str],
) -> str:
    lines = [
        f"# Raw diary text: {week_id}",
        "",
        f"Source PDF: {_source_pdf_display_path(pdf_path)}",
        "",
    ]

    for index, text in enumerate(pages_text, start=1):
        lines.append(f"<!-- page {index} -->")
        lines.append("")
        if text.strip():
            lines.append(text.rstrip())
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def raw_text_output_path(week_id: str, output_dir: Path | None = None) -> Path:
    base = output_dir or INTERIM_RAW_TEXT_DIR
    return base / f"{week_id}_raw.md"


def extract_pdf_to_raw_markdown(
    pdf_path: Path | str,
    week_id: str,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Extract PDF text and save it as interim raw markdown."""
    week_id = _validate_week_id(week_id)
    pdf_path = _validate_pdf_path(Path(pdf_path))
    pages_text = _extract_pages_text(pdf_path)
    content = _format_raw_markdown(week_id, pdf_path, pages_text)

    out_dir = output_dir or INTERIM_RAW_TEXT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_text_output_path(week_id, out_dir)
    output_path.write_text(content, encoding="utf-8")
    return output_path
