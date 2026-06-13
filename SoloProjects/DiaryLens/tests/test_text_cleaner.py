from pathlib import Path

import pytest

from diarylens.text_cleaner import (
    TextCleaningError,
    clean_raw_markdown,
    clean_text_content,
    clean_text_output_path,
    raw_text_input_path,
)


def _consecutive_blank_run(lines: list[str]) -> list[int]:
    runs: list[int] = []
    current = 0
    for line in lines:
        if line == "":
            current += 1
        else:
            if current:
                runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs or [0]


SAMPLE_RAW = """# Raw diary text: 2026-W22

Source PDF: data/raw/weekly/week.pdf


<!-- page 1 -->

31 мая 2026 г. Вс   
В поисках веселья


<!-- page 2 -->

30 мая 2026 г. Сб
"""


def test_clean_text_creates_clean_markdown(tmp_path):
    raw_dir = tmp_path / "raw_text"
    clean_dir = tmp_path / "clean_text"
    raw_dir.mkdir()
    raw_path = raw_text_input_path("2026-W22", raw_dir)
    raw_path.write_text(SAMPLE_RAW, encoding="utf-8")

    output_path = clean_raw_markdown(
        "2026-W22",
        input_dir=raw_dir,
        output_dir=clean_dir,
    )

    assert output_path == clean_text_output_path("2026-W22", clean_dir)
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("# Clean diary text: 2026-W22\n")
    assert "Source PDF: data/raw/weekly/week.pdf" in content
    assert "<!-- page 1 -->" in content
    assert "<!-- page 2 -->" in content
    assert "31 мая 2026 г. Вс" in content
    assert "30 мая 2026 г. Сб" in content


def test_clean_text_normalizes_whitespace_and_blank_lines():
    raw = (
        "# Raw diary text: 2026-W22\n"
        "\n"
        "\n"
        "\n"
        "Line with trailing spaces   \n"
        "\n"
        "\n"
        "\n"
        "\n"
        "Another line\n"
        "\n"
        "\n"
    )

    content = clean_text_content(raw, "2026-W22")
    lines = content.split("\n")

    assert "Line with trailing spaces" in lines
    assert "Another line" in lines
    assert not any(line.endswith("   ") for line in lines if line)
    assert max(_consecutive_blank_run(lines)) <= 2
    assert content.startswith("# Clean diary text: 2026-W22\n")
    assert content.endswith("Another line\n")
    assert lines[0] != ""


def test_clean_text_preserves_page_markers():
    raw = (
        "# Raw diary text: 2026-W22\n"
        "\n"
        "<!-- page 1 -->\n"
        "\n"
        "Entry one\n"
        "\n"
        "<!-- page 2 -->\n"
        "\n"
        "Entry two\n"
    )

    content = clean_text_content(raw, "2026-W22")

    assert "<!-- page 1 -->" in content
    assert "<!-- page 2 -->" in content
    assert content.index("<!-- page 1 -->") < content.index("Entry one")
    assert content.index("Entry one") < content.index("<!-- page 2 -->")
    assert content.index("<!-- page 2 -->") < content.index("Entry two")


def test_clean_text_missing_raw_file_raises(tmp_path):
    with pytest.raises(TextCleaningError, match="Raw markdown not found"):
        clean_raw_markdown("2026-W22", input_dir=tmp_path, output_dir=tmp_path)


def test_clean_text_requires_week_id():
    with pytest.raises(TextCleaningError, match="week_id is required"):
        clean_raw_markdown("")


def test_clean_text_empty_raw_raises(tmp_path):
    raw_dir = tmp_path / "raw_text"
    raw_dir.mkdir()
    raw_path = raw_text_input_path("2026-W22", raw_dir)
    raw_path.write_text("   \n", encoding="utf-8")

    with pytest.raises(TextCleaningError, match="Raw markdown is empty"):
        clean_raw_markdown("2026-W22", input_dir=raw_dir, output_dir=tmp_path)
