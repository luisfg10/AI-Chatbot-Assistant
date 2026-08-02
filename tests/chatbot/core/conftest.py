import json
from collections.abc import Callable
from types import SimpleNamespace

import pytest

# ------------------------------------------------------------------
# Shared fixtures for building fake OpenAI chat completion objects.
# pytest auto-discovers `conftest.py` files, making these fixtures
# available to every test module in this directory (e.g. test_agent.py).


@pytest.fixture
def models_dict() -> dict[str, dict[str, str]]:
    """Build a minimal models dictionary for agent initialization."""
    return {
        "model-a": {
            "base url": "https://example-a.test/v1",
            "api key": "test-key-a"
        },
        "model-b": {
            "base url": "https://example-b.test/v1",
            "api key": "test-key-b"
        }
    }


@pytest.fixture
def make_tool_call() -> Callable[..., SimpleNamespace]:
    """Build a factory for fake OpenAI tool call objects."""
    def _make(
        tool_name: str,
        tool_call_id: str = "tool-call-1",
        tool_type: str = "function",
        arguments: dict | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=tool_call_id,
            type=tool_type,
            function=SimpleNamespace(
                name=tool_name,
                arguments=json.dumps(arguments or {})
            )
        )
    return _make


@pytest.fixture
def make_message() -> Callable[..., SimpleNamespace]:
    """Build a factory for fake OpenAI assistant messages."""
    def _make(
        role: str = "assistant",
        content: str | None = None,
        tool_calls: list | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(role=role, content=content, tool_calls=tool_calls)
    return _make


@pytest.fixture
def make_response() -> Callable[..., SimpleNamespace]:
    """Build a factory for fake OpenAI chat completion responses."""
    def _make(finish_reason: str, message: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish_reason, message=message)]
        )
    return _make
