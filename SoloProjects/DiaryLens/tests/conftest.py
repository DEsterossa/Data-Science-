import pytest


@pytest.fixture(autouse=True)
def _default_gigachat_model_env(monkeypatch):
    """Provide model env vars for tests unless a test overrides them."""
    monkeypatch.setenv("GIGACHAT_MODEL_DAILY", "test-daily-model")
    monkeypatch.setenv("GIGACHAT_MODEL_WEEKLY", "test-weekly-model")
    monkeypatch.setenv("GIGACHAT_MODEL_REPORT", "test-report-model")
