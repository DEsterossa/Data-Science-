import json
from pathlib import Path

import numpy as np
import pytest

from diarylens.ask_history.models import AskHistoryConfig, AskHistoryDebug, AskHistoryError
from diarylens.ask_history.pipeline import ask_history


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "скука" in lower:
                vectors.append(np.array([1.0, 0.0], dtype=np.float32))
            else:
                vectors.append(np.array([0.0, 1.0], dtype=np.float32))
        return np.vstack(vectors)


def _setup_project(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    days_json_dir = project_root / "data" / "processed" / "days_json"
    days_md_dir = project_root / "data" / "processed" / "days_md"
    memory_dir = project_root / "data" / "memory"
    prompts_dir = project_root / "prompts"
    for path in (days_json_dir, days_md_dir, memory_dir, prompts_dir):
        path.mkdir(parents=True)

    monkeypatch.setattr("diarylens.config.PROJECT_ROOT", project_root)
    monkeypatch.setattr("diarylens.config.DAYS_JSON_DIR", days_json_dir)
    monkeypatch.setattr("diarylens.config.DAYS_MD_DIR", days_md_dir)
    monkeypatch.setattr("diarylens.config.MEMORY_DIR", memory_dir)
    monkeypatch.setattr("diarylens.config.PROMPTS_DIR", prompts_dir)

    payload = {
        "date": "2026-06-12",
        "week_id": "2026-W24",
        "source_day_md": "data/processed/days_md/2026-06-12.md",
        "important_moments": [{"quote": "Была скука", "note": "настроение"}],
        "short_summary": "День, где была скука.",
    }
    (days_json_dir / "2026-06-12.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (days_md_dir / "2026-06-12.md").write_text(
        "# Diary day: 2026-06-12\n\nСегодня была скука и прогулка по городу.",
        encoding="utf-8",
    )
    (prompts_dir / "ask_history.md").write_text(
        "question={question}\nevidence={evidence_context}",
        encoding="utf-8",
    )
    return memory_dir


def test_ask_history_returns_answer_string(tmp_path, monkeypatch):
    memory_dir = _setup_project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "diarylens.ask_history.pipeline.load_embedding_model",
        lambda model_name: FakeEmbeddingModel(),
    )
    monkeypatch.setattr(
        "diarylens.ask_history.pipeline.generate_text",
        lambda prompt, model=None: "# Ответ\n\nНашёл evidence.",
    )

    answer = ask_history(
        "Когда я чувствую скуку?",
        config=AskHistoryConfig(top_k_days=1, top_k_chunks=1, day_header_skip_chars=0),
        force_rebuild_index=True,
    )

    assert "# Ответ" in answer
    assert (memory_dir / "day_index.csv").exists()
    assert (memory_dir / "day_embeddings_cache.csv").exists()


def test_ask_history_saves_markdown_answer(tmp_path, monkeypatch):
    memory_dir = _setup_project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "diarylens.ask_history.pipeline.load_embedding_model",
        lambda model_name: FakeEmbeddingModel(),
    )
    monkeypatch.setattr(
        "diarylens.ask_history.pipeline.generate_text",
        lambda prompt, model=None: "# Ответ\n\nСохранённый markdown.",
    )

    answer = ask_history(
        "Когда я чувствую скуку?",
        config=AskHistoryConfig(top_k_days=1, top_k_chunks=1, day_header_skip_chars=0),
        force_rebuild_index=True,
    )

    answer_paths = list((memory_dir / "ask_history_answers").glob("*.md"))
    assert len(answer_paths) == 1
    assert answer_paths[0].read_text(encoding="utf-8") == answer + "\n"


def test_ask_history_return_debug_contains_prompt_and_answer(tmp_path, monkeypatch):
    _setup_project(tmp_path, monkeypatch)
    captured = {}
    monkeypatch.setattr(
        "diarylens.ask_history.pipeline.load_embedding_model",
        lambda model_name: FakeEmbeddingModel(),
    )

    def fake_generate(prompt, model=None):
        captured["prompt"] = prompt
        return "# Ответ\n\nDebug answer."

    monkeypatch.setattr("diarylens.ask_history.pipeline.generate_text", fake_generate)

    debug = ask_history(
        "Когда я чувствую скуку?",
        config=AskHistoryConfig(top_k_days=1, top_k_chunks=1, day_header_skip_chars=0),
        return_debug=True,
        force_rebuild_index=True,
    )

    assert isinstance(debug, AskHistoryDebug)
    assert "Когда я чувствую скуку?" in debug.prompt
    assert "Сегодня была скука" in debug.prompt
    assert captured["prompt"] == debug.prompt
    assert debug.answer == "# Ответ\n\nDebug answer."
    assert debug.answer_path is not None
    assert Path(debug.answer_path).read_text(encoding="utf-8") == debug.answer + "\n"


def test_force_rebuild_index_recreates_day_index(tmp_path, monkeypatch):
    memory_dir = _setup_project(tmp_path, monkeypatch)
    (memory_dir / "day_index.csv").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        "diarylens.ask_history.pipeline.load_embedding_model",
        lambda model_name: FakeEmbeddingModel(),
    )
    monkeypatch.setattr(
        "diarylens.ask_history.pipeline.generate_text",
        lambda prompt, model=None: "# Ответ",
    )

    ask_history(
        "скука",
        config=AskHistoryConfig(top_k_days=1, top_k_chunks=1, day_header_skip_chars=0),
        force_rebuild_index=True,
    )

    assert "2026-06-12" in (memory_dir / "day_index.csv").read_text(encoding="utf-8")


def test_missing_days_json_raises_clear_error(tmp_path, monkeypatch):
    project_root = tmp_path
    days_json_dir = project_root / "data" / "processed" / "days_json"
    days_md_dir = project_root / "data" / "processed" / "days_md"
    memory_dir = project_root / "data" / "memory"
    prompts_dir = project_root / "prompts"
    for path in (days_json_dir, days_md_dir, memory_dir, prompts_dir):
        path.mkdir(parents=True)
    monkeypatch.setattr("diarylens.config.PROJECT_ROOT", project_root)
    monkeypatch.setattr("diarylens.config.DAYS_JSON_DIR", days_json_dir)
    monkeypatch.setattr("diarylens.config.DAYS_MD_DIR", days_md_dir)
    monkeypatch.setattr("diarylens.config.MEMORY_DIR", memory_dir)
    monkeypatch.setattr("diarylens.config.PROMPTS_DIR", prompts_dir)

    with pytest.raises(AskHistoryError, match="no daily JSON files found"):
        ask_history("скука", force_rebuild_index=True)
