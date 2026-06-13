from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from diarylens.pdf_extractor import (
    PdfExtractionError,
    extract_pdf_to_raw_markdown,
    raw_text_output_path,
)


def _mock_page(text: str) -> MagicMock:
    page = MagicMock()
    page.extract_text.return_value = text
    return page


@patch("diarylens.pdf_extractor.PdfReader")
def test_extract_pdf_creates_raw_markdown(mock_reader, tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    pdf_path = project_root / "data" / "raw" / "weekly" / "2026-W23.pdf"
    output_dir = project_root / "data" / "interim" / "raw_text"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr("diarylens.pdf_extractor.PROJECT_ROOT", project_root)

    reader = MagicMock()
    reader.pages = [
        _mock_page("Monday diary entry"),
        _mock_page("Tuesday diary entry"),
    ]
    mock_reader.return_value = reader

    output_path = extract_pdf_to_raw_markdown(
        pdf_path,
        "2026-W23",
        output_dir=output_dir,
    )

    assert output_path == raw_text_output_path("2026-W23", output_dir)
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("# Raw diary text: 2026-W23\n")
    assert "Source PDF: data/raw/weekly/2026-W23.pdf" in content
    assert "<!-- page 1 -->" in content
    assert "<!-- page 2 -->" in content
    assert "Monday diary entry" in content
    assert "Tuesday diary entry" in content
    assert content.index("<!-- page 1 -->") < content.index("<!-- page 2 -->")
    assert content.index("Monday diary entry") < content.index("Tuesday diary entry")


def test_extract_pdf_missing_file_raises(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(PdfExtractionError, match="PDF file not found"):
        extract_pdf_to_raw_markdown(missing_pdf, "2026-W23", output_dir=tmp_path)


def test_extract_pdf_requires_week_id(tmp_path):
    pdf_path = tmp_path / "week.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with pytest.raises(PdfExtractionError, match="week_id is required"):
        extract_pdf_to_raw_markdown(pdf_path, "", output_dir=tmp_path)
