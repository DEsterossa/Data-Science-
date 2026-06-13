"""Minimal GigaChat LLM client for DiaryLens."""

import os

from dotenv import load_dotenv
from gigachat import GigaChat

from diarylens.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env", override=True)


class LLMError(Exception):
    """Raised when LLM generation fails."""


TEST_LLM_PROMPT = 'Верни только JSON: {"ok": true}'

MODEL_KINDS = ("daily", "weekly", "report")

MODEL_KIND_ENV_VARS = {
    "daily": "GIGACHAT_MODEL_DAILY",
    "weekly": "GIGACHAT_MODEL_WEEKLY",
    "report": "GIGACHAT_MODEL_REPORT",
}

DEFAULT_GIGACHAT_TIMEOUT = 120.0


def _get_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _get_auth_key() -> str:
    auth_key = os.environ.get("GIGACHAT_AUTH_KEY")
    if not auth_key:
        raise LLMError("GIGACHAT_AUTH_KEY environment variable is not set")
    return auth_key


def _get_verify_ssl_certs() -> bool:
    value = _get_env("GIGACHAT_VERIFY_SSL_CERTS", "false").lower()
    return value in ("1", "true", "yes")


def _get_timeout() -> float:
    raw = os.environ.get("GIGACHAT_TIMEOUT")
    if raw is None or not raw.strip():
        return DEFAULT_GIGACHAT_TIMEOUT
    try:
        return float(raw)
    except ValueError as exc:
        raise LLMError("Invalid GIGACHAT_TIMEOUT: expected float seconds.") from exc


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _require_env(name: str) -> str:
    value = _optional_env(name)
    if value is None:
        raise LLMError(f"{name} environment variable is not set")
    return value


def _build_gigachat_kwargs(auth_key: str, scope: str, model_name: str) -> dict:
    kwargs = {
        "credentials": auth_key,
        "scope": scope,
        "model": model_name,
        "verify_ssl_certs": _get_verify_ssl_certs(),
        "timeout": _get_timeout(),
    }
    ca_bundle_file = os.environ.get("GIGACHAT_CA_BUNDLE_FILE")
    if ca_bundle_file:
        kwargs["ca_bundle_file"] = ca_bundle_file
    return kwargs


def resolve_model_for_kind(kind: str) -> str:
    """Resolve a configured GigaChat model name for a pipeline role."""
    if kind not in MODEL_KIND_ENV_VARS:
        raise LLMError(
            f"Unknown model kind: {kind}. Expected one of: {', '.join(MODEL_KINDS)}"
        )
    return _require_env(MODEL_KIND_ENV_VARS[kind])


def resolve_report_model() -> str:
    """Resolve report model from GIGACHAT_MODEL_REPORT, else GIGACHAT_MODEL_WEEKLY."""
    report_model = _optional_env("GIGACHAT_MODEL_REPORT")
    if report_model is not None:
        return report_model
    return resolve_model_for_kind("weekly")


def generate_text(prompt: str, model: str | None = None) -> str:
    """Send a prompt to GigaChat and return the response text."""
    auth_key = _get_auth_key()
    scope = _get_env("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    model_name = model or resolve_model_for_kind("daily")

    try:
        with GigaChat(**_build_gigachat_kwargs(auth_key, scope, model_name)) as client:
            response = client.chat(prompt)
    except Exception as exc:
        raise LLMError(f"GigaChat request failed: {exc}") from exc

    content = response.choices[0].message.content
    if not content:
        raise LLMError("GigaChat returned an empty response")
    return content


def run_test_llm(model_kind: str) -> str:
    """Run a short connectivity test against the selected GigaChat model."""
    if model_kind == "report":
        model_name = resolve_report_model()
    else:
        model_name = resolve_model_for_kind(model_kind)
    return generate_text(TEST_LLM_PROMPT, model=model_name)
