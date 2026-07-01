"""Ask-history retrieval and answer generation pipeline."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from diarylens import config as project_config
from diarylens.ask_history.chunking import build_local_chunks_from_days
from diarylens.ask_history.day_index import (
    build_day_index,
    load_day_index_csv,
    save_day_index_csv,
)
from diarylens.ask_history.embeddings import (
    build_or_load_day_embeddings,
    load_embedding_model,
)
from diarylens.ask_history.models import AskHistoryConfig, AskHistoryDebug, AskHistoryError
from diarylens.ask_history.prompt import (
    build_evidence_context,
    load_prompt_template,
    render_ask_history_prompt,
)
from diarylens.ask_history.retrieval import search_days_embeddings, search_evidence_chunks
from diarylens.llm_client import LLMError, generate_text, resolve_report_model


def _day_index_path():
    return project_config.MEMORY_DIR / "day_index.csv"


def _day_embeddings_cache_path():
    return project_config.MEMORY_DIR / "day_embeddings_cache.csv"


def _ask_history_prompt_path():
    return project_config.PROMPTS_DIR / "ask_history.md"


def _ask_history_answers_dir() -> Path:
    return project_config.MEMORY_DIR / "ask_history_answers"


def save_ask_history_answer(question: str, answer: str) -> Path:
    """Save the generated markdown answer to data/memory."""
    answers_dir = _ask_history_answers_dir()
    answers_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:10]
    output_path = answers_dir / f"{created_at}_{question_hash}.md"
    output_path.write_text(answer.rstrip() + "\n", encoding="utf-8")
    return output_path


def _load_or_build_day_index(
    config: AskHistoryConfig,
    *,
    force_rebuild_index: bool,
):
    index_path = _day_index_path()
    if force_rebuild_index or not index_path.exists():
        records = build_day_index(
            project_config.DAYS_JSON_DIR,
            project_config.DAYS_MD_DIR,
            config,
        )
        save_day_index_csv(records, index_path)
        return records
    return load_day_index_csv(index_path)


def _run_retrieval(
    question: str,
    config: AskHistoryConfig,
    *,
    force_rebuild_index: bool,
) -> AskHistoryDebug:
    project_config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_or_build_day_index(
        config,
        force_rebuild_index=force_rebuild_index,
    )
    model = load_embedding_model(config.model_name)
    embeddings = build_or_load_day_embeddings(
        records,
        _day_embeddings_cache_path(),
        config.model_name,
        model=model,
    )
    day_results = search_days_embeddings(
        question,
        records,
        embeddings,
        model,
        config.top_k_days,
    )
    chunks = build_local_chunks_from_days(day_results, config)
    evidence_results = search_evidence_chunks(
        question,
        chunks,
        model,
        config.top_k_chunks,
    )
    evidence_context = build_evidence_context(
        evidence_results,
        config.max_chunk_chars,
    )
    return AskHistoryDebug(
        question=question,
        config=config,
        day_results=day_results,
        evidence_results=evidence_results,
        evidence_context=evidence_context,
        prompt="",
        answer=None,
    )


def search_memory(
    question: str,
    config: AskHistoryConfig | None = None,
    force_rebuild_index: bool = False,
) -> AskHistoryDebug:
    """Run retrieval only and return day/chunk debug information."""
    active_config = config or AskHistoryConfig()
    return _run_retrieval(
        question,
        active_config,
        force_rebuild_index=force_rebuild_index,
    )


def ask_history(
    question: str,
    config: AskHistoryConfig | None = None,
    return_debug: bool = False,
    force_rebuild_index: bool = False,
) -> str | AskHistoryDebug:
    """Answer a question using raw day markdown evidence."""
    active_config = config or AskHistoryConfig()
    debug = _run_retrieval(
        question,
        active_config,
        force_rebuild_index=force_rebuild_index,
    )
    prompt_template = load_prompt_template(_ask_history_prompt_path())
    prompt = render_ask_history_prompt(
        prompt_template,
        question=question,
        evidence_context=debug.evidence_context,
    )

    try:
        answer = generate_text(prompt, model=resolve_report_model())
    except LLMError as exc:
        raise AskHistoryError(f"ask-history LLM request failed: {exc}") from exc

    answer_path = save_ask_history_answer(question, answer)

    if return_debug:
        return debug.model_copy(
            update={
                "prompt": prompt,
                "answer": answer,
                "answer_path": answer_path.as_posix(),
            }
        )
    return answer
