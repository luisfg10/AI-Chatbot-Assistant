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

        Should match src/chatbot/context/system.yaml.
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
                supported_personalities="hello bob"
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
                available_models="hello bob",
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
        default_config_dict["personality"] = "weird"

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
        assert chatbot.messages[0]["content"] == chatbot.get_chatbot_instructions("professional")

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
    # chatbot_call

    def test_chatbot_call_appends_messages_and_returns_response(
        self,
        chatbot: ChatbotAssistant,
        mocker: Any
    ) -> None:
        """Test a call appends the user/assistant messages and returns the response text."""
        # Mock LLM API call
        mocker.patch.object(
            ChatCompletionsBaseAgent,
            "llm_api_call",
            return_value=[{"role": "assistant", "content": "Hi there!"}]
        )

        response = chatbot.chatbot_call("Hello")

        # Check returned response
        assert response == "Hi there!"

        # Check messages live in self
        assert chatbot.messages[-2] == {"role": "user", "content": "Hello"}
        assert chatbot.messages[-1] == {"role": "assistant", "content": "Hi there!"}

    # ------------------------------------------------------------------
    # _compact_messages

    def test_compact_messages_noop_below_limit(
        self,
        chatbot: ChatbotAssistant
    ) -> None:
        """Test messages are left untouched when below the compacting limit."""
        chatbot.messages = [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "Hi"}
        ]
        original_messages = list(chatbot.messages)

        chatbot._compact_messages()  # Should include escape logic if below limit

        assert chatbot.messages == original_messages

    def test_compact_messages_summarizes_and_trims_when_above_limit(
        self,
        chatbot: ChatbotAssistant,
        mocker: Any
    ) -> None:
        """Test messages are summarized and trimmed to instructions + summary when above the limit."""
        # Long messages list that triggers compacting
        chatbot.messages = [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "It's sunny."}
        ]
        # Patch next LLM API call, which should be for compacting.
        mocker.patch.object(
            ChatCompletionsBaseAgent,
            "llm_api_call",
            return_value=[{"role": "assistant", "content": "User asked about weather."}]
        )

        # Run compacting
        chatbot._compact_messages()

        # Check compacting ran successfully
        assert len(chatbot.messages) == 2
        assert chatbot.messages[0] == {"role": "system", "content": "instructions"}
        assert "User asked about weather." in chatbot.messages[1]["content"]
        assert chatbot.long_term_memory == [
            {"role": "assistant", "content": "User asked about weather."}
        ]

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
