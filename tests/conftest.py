"""Fixtures needed for running the entire test suite."""
import pytest


@pytest.fixture(autouse=True)
def mock_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Set dummy env var for OpenAI API key.

    OpenAI's chat completions client requires passing an API key
    on client init, so this is meant to avoid that error on CI
    since it runs without a .env file.

    Parameters
    ----------
        monkeypatch: pytest.MonkeyPatch
            Built-in pytest fixture.
            Used for specifying behaviors at runtime.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-for-tests")
