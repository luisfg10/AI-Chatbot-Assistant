from typing import Any

import pytest

from src.chatbot.core.agent import ChatbotAssistant
from src.chatbot.core.base_chat_completions import ChatCompletionsBaseAgent


class TestChatbotAssistant:
    """Test `ChatbotAssistant`."""

    # ------------------------------------------------------------------
    # Fixtures

    @pytest.fixture
    def available_models_dict(self) -> dict[str, dict]:
        """Build a minimal available-models dict for class init."""
        return {
            "provider-a": {
                "available models": ["model-a"],
                "urls": {"api": "https://example-a.test/v1"}
            }
        }

    @pytest.fixture
    def provider_api_keys_dict(self) -> dict[str, str]:
        """Build a matching API key dict for `available_models_dict`."""
        return {"provider-a": "test-key-a"}

    @pytest.fixture
    def default_config_dict(self) -> dict:
        """Build a minimal default config dict for class init."""
        return {
            "model": "model-a",
            "personality": "friendly",
            "compacting message limit": 4,
            "max recursive tool calls": 5,
            "max completion tokens": 100
        }

    @pytest.fixture
    def supported_personalities_list(self) -> list[str]:
        """
        Build a supported personalities list.

        Should exist within src/chatbot/context/system.yaml.
        """
        return ["friendly", "professional"]

    @pytest.fixture
    def chatbot(
        self,
        available_models_dict: dict[str, dict],
        provider_api_keys_dict: dict[str, str],
        default_config_dict: dict,
        supported_personalities_list: list[str]
    ) -> ChatbotAssistant:
        """Build a default ChatbotAssistant instance for tests."""
        return ChatbotAssistant(
            available_models=available_models_dict,
            provider_api_keys=provider_api_keys_dict,
            default_config=default_config_dict,
            supported_personalities=supported_personalities_list
        )

    # ------------------------------------------------------------------
    # __init__

    def test_init_sets_default_personality_and_instructions(
        self,
        chatbot: ChatbotAssistant
    ) -> None:
        """Test init sets the default personality and system message."""
        # Personality (as specified in the fixture)
        assert chatbot.default_personality == "friendly"

        # Check agent instructions are set as first message
        assert chatbot.messages[0]["role"] == "system"
        assert len(chatbot.messages[0]["content"]) > 0

    def test_init_invalid_supported_personalities_raises(
        self,
        available_models_dict: dict[str, dict],
        provider_api_keys_dict: dict[str, str],
        default_config_dict: dict
    ) -> None:
        """Test an empty or invalid `supported_personalities` raises ValueError."""
        # Empty list
        with pytest.raises(ValueError, match="supported_personalities"):
            ChatbotAssistant(
                available_models=available_models_dict,
                provider_api_keys=provider_api_keys_dict,
                default_config=default_config_dict,
                supported_personalities=[]
            )

        # Invalid object
        with pytest.raises(ValueError, match="supported_personalities"):
            ChatbotAssistant(
                available_models=available_models_dict,
                provider_api_keys=provider_api_keys_dict,
                default_config=default_config_dict,
                supported_personalities="sassy"
            )

    def test_init_invalid_available_models_raises(
        self,
        provider_api_keys_dict: dict[str, str],
        default_config_dict: dict,
        supported_personalities_list: list[str]
    ) -> None:
        """Test an empty or invalid `available_models` raises ValueError."""
        # Empty
        with pytest.raises(ValueError, match="available_models"):
            ChatbotAssistant(
                available_models={},
                provider_api_keys=provider_api_keys_dict,
                default_config=default_config_dict,
                supported_personalities=supported_personalities_list
            )

        # Invalid
        with pytest.raises(ValueError, match="available_models"):
            ChatbotAssistant(
                available_models="cortana",
                provider_api_keys=provider_api_keys_dict,
                default_config=default_config_dict,
                supported_personalities=supported_personalities_list
            )

    def test_init_falls_back_to_first_personality_when_default_invalid(
        self,
        available_models_dict: dict[str, dict],
        provider_api_keys_dict: dict[str, str],
        default_config_dict: dict,
        supported_personalities_list: list[str]
    ) -> None:
        """Test an unsupported default personality falls back to the first supported one."""
        # Set invalid personality
        default_config_dict["personality"] = "icky"

        # Init class
        chatbot = ChatbotAssistant(
            available_models=available_models_dict,
            provider_api_keys=provider_api_keys_dict,
            default_config=default_config_dict,
            supported_personalities=supported_personalities_list
        )
        # Check personality is first of valid list
        assert chatbot.default_personality == supported_personalities_list[0]

    # ------------------------------------------------------------------
    # set_personality

    def test_set_personality_updates_instructions(
        self,
        chatbot: ChatbotAssistant
    ) -> None:
        """Test switching personality changes the stored system message."""
        # Save original instructions
        original_content = chatbot.messages[0]["content"]

        # Change personality
        chatbot.set_personality("professional")

        # Validate that new instructions were saved to self
        assert chatbot.messages[0]["content"] != original_content
        assert chatbot.messages[0]["content"] == chatbot.get_agent_instructions("professional")

    def test_set_personality_invalid_falls_back_to_default(
        self,
        chatbot: ChatbotAssistant
    ) -> None:
        """Test an unsupported personality falls back to the current default's instructions."""
        original_content = chatbot.messages[0]["content"]
        # Set invalid personality
        chatbot.set_personality("politician")

        # Check instructions remain the same
        assert chatbot.messages[0]["content"] == original_content

    # ------------------------------------------------------------------
    # set_tools

    def test_set_tools_default_includes_all_available_tools(
        self,
        chatbot: ChatbotAssistant
    ) -> None:
        """Test init sets all registered tools available by default."""
        assert set(
            chatbot.tool_registry.keys()) == set(chatbot.available_tools)
        assert len(chatbot.tool_schema) == len(chatbot.available_tools)

    def test_set_tools_with_tool_names_filters_schema_and_registry(
        self,
        chatbot: ChatbotAssistant
    ) -> None:
        """Test `tool_names` restricts the active tool schema and registry."""
        tool_name = chatbot.available_tools[0]

        chatbot.set_tools(tool_names=[tool_name])

        assert list(chatbot.tool_registry.keys()) == [tool_name]
        assert [item["function"]["name"] for item in chatbot.tool_schema] == [tool_name]
        assert len(chatbot.available_tools) >= 1

    def test_set_tools_empty_list_disables_all_tools(
        self,
        chatbot: ChatbotAssistant
    ) -> None:
        """Test an explicit empty `tool_names` list disables every tool."""
        chatbot.set_tools(tool_names=[])

        assert chatbot.tool_registry == {}
        assert chatbot.tool_schema == []
        assert len(chatbot.available_tools) > 0

    def test_set_tools_unknown_tool_name_ignored(
        self,
        chatbot: ChatbotAssistant
    ) -> None:
        """Test a name not present in the registry results in no active tools."""
        chatbot.set_tools(tool_names=["bob the builder's hammer"])

        assert chatbot.tool_registry == {}
        assert chatbot.tool_schema == []

    # ------------------------------------------------------------------
    # Agent call

    def test_agent_call_appends_messages_and_returns_response(
        self,
        chatbot: ChatbotAssistant,
        mocker: Any
    ) -> None:
        """Test a call appends messages and returns the response text."""
        # Mock LLM API call
        mocker.patch.object(
            ChatCompletionsBaseAgent,
            "llm_api_call",
            return_value=[{"role": "assistant", "content": "Hi there!"}]
        )

        response = chatbot("Hello")

        # Check returned response
        assert response == "Hi there!"

        # Check both user and system messages were saved
        assert chatbot.messages[-2] == {"role": "user", "content": "Hello"}
        assert chatbot.messages[-1] == {"role": "assistant", "content": "Hi there!"}

    # ------------------------------------------------------------------
    # _compact_messages

    def test_compact_messages_noop_below_limit(
        self,
        chatbot: ChatbotAssistant
    ) -> None:
        """Test messages are left unchanged when below the compacting limit."""
        chatbot.messages = [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "Hi"}
        ]
        original_messages = list(chatbot.messages)

        # Assumption: compacting method includes escape logic if below limit
        chatbot._compact_messages()
        assert chatbot.messages == original_messages

    def test_compact_messages_summarizes_and_trims_when_above_limit(
        self,
        chatbot: ChatbotAssistant,
        mocker: Any
    ) -> None:
        """Test messages are summarized when above the limit."""
        # Messgaes list above config fixture limit (5 vs. 4)
        # should trigger compacting
        chatbot.messages = [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "It's sunny."}
        ]
        # Patch next LLM API call, mimicking compacting summary
        mocker.patch.object(
            ChatCompletionsBaseAgent,
            "llm_api_call",
            return_value=[{
                "role": "assistant",
                "content": "User asked about weather."
            }]
        )
        # Run compacting
        chatbot._compact_messages()

        # Check assertions
        assert len(chatbot.messages) == 2
        assert chatbot.messages[0] == {
            "role": "system",
            "content": "instructions"
        }
        assert "User asked about weather." in chatbot.messages[1]["content"]
        assert chatbot.long_term_memory == [{
            "role": "assistant",
            "content": "User asked about weather."
        }]

    # ------------------------------------------------------------------
    # reset_memory

    def test_reset_memory_keeps_only_instructions(
        self,
        chatbot: ChatbotAssistant
    ) -> None:
        """Test resetting memory keeps only the original instructions message."""
        instructions_message = chatbot.messages[0]
        chatbot.messages.append({"role": "user", "content": "Hi"})
        chatbot.messages.append({"role": "assistant", "content": "Hello!"})

        chatbot.reset_memory()

        assert chatbot.messages == [instructions_message]
