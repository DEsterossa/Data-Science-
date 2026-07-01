"""Prompt construction for ask-history answers."""

from pathlib import Path

from diarylens.ask_history.models import EvidenceSearchResult


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def build_evidence_context(
    evidence_results: list[EvidenceSearchResult],
    max_chunk_chars: int,
) -> str:
    """Render retrieved evidence chunks for the answer prompt."""
    blocks: list[str] = []
    for result in evidence_results:
        lines = [
            f"[{result.rank}] date: {result.date}",
        ]
        if result.week_id:
            lines.append(f"week_id: {result.week_id}")
        lines.extend(
            [
                f"score: {result.score:.4f}",
                f"source_day_md: {result.source_day_md}",
                f"chunk_index: {result.chunk_index}",
                "",
                _truncate(result.text, max_chunk_chars),
            ]
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def load_prompt_template(path: Path) -> str:
    """Load the ask-history markdown prompt template."""
    return path.read_text(encoding="utf-8")


def render_ask_history_prompt(
    template: str,
    *,
    question: str,
    evidence_context: str,
) -> str:
    """Fill the two ask-history prompt placeholders."""
    return (
        template.replace("{question}", question).replace(
            "{evidence_context}",
            evidence_context,
        )
    )
