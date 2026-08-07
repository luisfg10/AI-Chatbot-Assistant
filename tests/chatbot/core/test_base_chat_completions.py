from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from src.chatbot.core.base_chat_completions import ChatCompletionsBaseAgent


class TestChatCompletionsBaseAgent:
    """Test `ChatCompletionsBaseAgent`."""

    @pytest.fixture
    def agent(
        self,
        models_dict: dict[str, dict[str, str]]
    ) -> ChatCompletionsBaseAgent:
        """
        Build a default agent instance for tests.

        Parameters
        ----------
            models_dict: dict
                Previously-defined fixture inside
                tests/chatbot/core/conftest.py that returns
                a sample models_dict.
        """
        return ChatCompletionsBaseAgent(
            models=models_dict,
            default_model="model-a"
        )

    @pytest.fixture
    def mock_create(
        self,
        agent: ChatCompletionsBaseAgent,
        mocker: Any
    ) -> Any:
        """
        Patch the agent's chat completions create method.

        Parameters
        ----------
            agent: ChatCompletionsBaseAgent
                Previously-defined fixture creating the agent instance
                whose `client.chat.completions.create` method is replaced.
            mocker: Any
                The `pytest-mock` fixture, a thin wrapper around
                `unittest.mock` that also handles automatically undoing
                any patches at the end of each test.

        Returns
        -------
            Any
                `MagicMock` object that now lives in place of
                `agent.client.chat.completions.create`.

        Notes
        -----
            `mocker.patch.object(target, attribute_name)` replaces the
            attribute `attribute_name` on `target` with a substitute and
            returns that substitute. No explicit replacement value is
            given here because when `new`/`return_value`/`side_effect`
            aren't passed, `patch.object` defaults to creating a
            `MagicMock()` automatically. That auto-created `MagicMock` is
            exactly what gets returned, which is then configured in each
            test, e.g.:

                mock_create.return_value = make_response(...)
                # or, for multiple sequential calls:
                mock_create.side_effect = [make_response(...), make_response(...)]

            Because `agent.client.chat.completions.create` is replaced
            directly on the real client instance, any code that calls
            `self.client.chat.completions.create(...)` inside
            `llm_api_call` transparently invokes this mock instead of
            making a real network request.
        """
        return mocker.patch.object(
            agent.client.chat.completions,
            "create"
        )

    def test_llm_api_call_returns_stop_response(
        self,
        agent: ChatCompletionsBaseAgent,
        mock_create: Any,
        make_message: Callable[..., SimpleNamespace],
        make_response: Callable[..., SimpleNamespace]
    ) -> None:
        """
        Test a mocked stop response is parsed into an assistant message.

        Basically tests that the API call method is being called with the
        correct parameters.

        Parameters
        ----------
            agent: ChatCompletionsBaseAgent
                Previously-defined fixture creating the agent instance.
            mock_create: Any
                Previously-created fixture that replaces
                `agent.client.chat.completions.create` so no real network
                call happens.
            make_message/make_response: Callable
                Build a fake object mimicking `response.choices[0].finish_reason`
                and `.message.content`, which `llm_api_call` reads directly.
                A `"stop"` finish reason takes the non-tool-call branch,
                wrapping `message.content` into a single assistant message
                and returning immediately.

        Notes
        -----
            make_message and make_response are defined as fixtures inside
            `tests_chatbot_core/conftest.py` and loaded automatically
            for all directories at the same level or below.
        """
        # Define mock LLM API response: a chat message without tool calls
        mock_create.return_value = make_response(
            "stop", make_message(content="Hello!")
        )

        # Call mocked LLM API method
        response = agent.llm_api_call(
            messages=[{
                "role": "user",
                "content": "Hi"
            }]
        )

        # Check response
        assert response == [{"role": "assistant", "content": "Hello!"}]

    def test_llm_tool_call_executes_registered_tool(
        self,
        make_tool_call: Callable[..., SimpleNamespace]
    ) -> None:
        """Test a registered tool call executes and returns a tool message."""
        # Fake known tool call order from LLM
        tool_call = make_tool_call("get_current_date")

        # Execute fake tool call (doesn't involve LLM)
        result = ChatCompletionsBaseAgent.llm_tool_call([tool_call])

        # Validate the tool call results' structure
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == tool_call.id

    def test_llm_api_call_resolves_tool_call_into_final_response(
        self,
        agent: ChatCompletionsBaseAgent,
        mock_create: Any,
        make_tool_call: Callable[..., SimpleNamespace],
        make_message: Callable[..., SimpleNamespace],
        make_response: Callable[..., SimpleNamespace],
        mocker: Any
    ) -> None:
        """
        Test a tool-call round trip recurses into a final text response.

        `llm_api_call` is recursive: when the API responds with a
        `"tool_calls"` finish reason, it runs the tool(s) and calls itself
        again with the tool result appended, until the API eventually
        responds with `"stop"`. This test drives that whole round trip
        with two fake API responses and a patched tool executor.

        Parameters
        ----------
            agent: ChatCompletionsBaseAgent
                Previously-defined fixture creating the agent instance.
            mock_create: Any
                Previously-created fixture that replaces
                `agent.client.chat.completions.create` so no real network
                call happens.
            make_tool_call/make_message/make_response: Callable
                Build fake objects mimicking the OpenAI response shape
                that `llm_api_call` reads directly.
            mocker: Any
                The `pytest-mock` fixture, used here to also patch
                `ChatCompletionsBaseAgent.llm_tool_call` so the fake tool
                call isn't actually executed.

        Notes
        -----
            `mock_create.side_effect` is set to a two-item list, so the
            first call to `agent.client.chat.completions.create` (made by
            the outer `llm_api_call`) returns the `"tool_calls"` response,
            and the second call (made by the recursive `llm_api_call`
            after handling the tool call) returns the `"stop"` response.
            This is what lets one test simulate two different, sequential
            API replies instead of always returning the same one.

            `llm_tool_call` is patched separately with a fixed
            `return_value` (not `side_effect`) because it only needs to
            be called once here, always returning the same fake tool
            message regardless of arguments.
        """
        tool_call = make_tool_call("get_current_date", tool_call_id="call-1")
        # Mimick the client's recursive tool-calling behavior
        mock_create.side_effect = [
            make_response("tool_calls", make_message(tool_calls=[tool_call])),
            make_response("stop", make_message(content="Done."))
        ]

        # Patch tool calling
        mocker.patch.object(
            ChatCompletionsBaseAgent,
            "llm_tool_call",
            return_value=[{
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "2026-08-02"
            }]
        )

        # Call patched method to get fake response
        response = agent.llm_api_call(
            messages=[{"role": "user", "content": "What is today's date?"}],
            tools=[{"type": "function", "function": {"name": "get_current_date"}}]
        )

        # Check messages list contains two calls, one of them a tool call
        assert response[-1] == {"role": "assistant", "content": "Done."}
        assert mock_create.call_count == 2

    def test_set_client_defaults_to_agent_default_model(
        self,
        agent: ChatCompletionsBaseAgent
    ) -> None:
        """Test set_client falls back to the agent's default model."""
        agent.set_client(model_code=None)
        assert agent.model_code == "model-a"

    def test_set_client_uses_given_model_code(
        self,
        agent: ChatCompletionsBaseAgent
    ) -> None:
        """Test set_client switches to the given model code."""
        agent.set_client(model_code="model-b")
        assert agent.model_code == "model-b"

    def test_serializes_message_with_tool_calls(
        self,
        make_tool_call: Callable[..., SimpleNamespace],
        make_message: Callable[..., SimpleNamespace]
    ) -> None:
        """Test a tool-call message serializes into an OpenAI-compatible dict."""
        tool_call = make_tool_call("get_current_date", tool_call_id="call-42")
        message = make_message(tool_calls=[tool_call])

        result = ChatCompletionsBaseAgent.serialize_chat_completions_response(message)

        # Validate the return object's structure
        assert result["role"] == "assistant"
        assert result["tool_calls"][0]["id"] == "call-42"
        assert result["tool_calls"][0]["function"]["name"] == "get_current_date"
