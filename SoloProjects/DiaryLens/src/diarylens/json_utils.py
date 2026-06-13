"""Shared helpers for parsing JSON from LLM responses."""

import json

REFUSAL_MARKERS = (
    "генеративные языковые модели не обладают",
    "не обладают собственным мнением",
    "разговоры на чувствительные темы",
    "чувствительными темами",
    "временно ограничены",
    "благодарим за понимание",
    "не могу помочь",
    "не могу выполнить",
    "я не могу",
    "отказ",
)


class LLMJsonParseError(Exception):
    """Raised when an LLM response cannot be parsed as JSON."""


def is_refusal_response(text: str) -> bool:
    """Detect typical GigaChat refusal / safety responses."""
    lowered = text.strip().lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def load_json_dict(text: str) -> dict:
    from json_repair import repair_json

    candidates = [text, repair_json(text)]
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(data, dict):
            return data
        raise LLMJsonParseError("LLM JSON must be an object")

    assert last_error is not None
    raise LLMJsonParseError(f"LLM returned invalid JSON: {last_error}") from last_error


def parse_llm_json_response(
    response_text: str,
    *,
    refusal_message: str = "GigaChat refused to process this request",
) -> dict:
    """Parse JSON from an LLM response, tolerating fences and minor JSON defects."""
    if not response_text.strip():
        raise LLMJsonParseError("LLM returned an empty response")

    if is_refusal_response(response_text):
        raise LLMJsonParseError(refusal_message)

    cleaned = extract_json_object(strip_markdown_fences(response_text))
    if not cleaned.strip():
        raise LLMJsonParseError(
            "LLM response does not contain a JSON object. "
            f"Response starts with: {response_text.strip()[:200]}"
        )
    return load_json_dict(cleaned)
