from loguru import logger

from config import AppConfig
from src.chatbot.core.base_chat_completions import ChatCompletionsBaseAgent
from src.chatbot.core.context import ChatbotContextHelper
from src.chatbot.tools import tool_schema


class ChatbotAssistant(ChatCompletionsBaseAgent, ChatbotContextHelper):
    """
    AI agent that interacts with a user by chat.

    Inherits from ChatCompletionsBaseAgent for LLM API interactions
    and from ChatbotContextHelper for managing the chatbot's context.
    """

    def __init__(
            self,
            available_models: dict = AppConfig.AVAILABLE_MODELS,
            provider_api_keys: dict = AppConfig.PROVIDER_API_KEYS,
            default_config: dict = AppConfig.DEFAULT_CONFIG,
            supported_personalities: list[str] | tuple[str] = AppConfig.SUPPORTED_CHATBOT_PERSONALITIES
    ) -> None:
        """
        Initialize the class instance.

        Parameters
        ----------
        available_models: dict
            A dict of available LLM providers and their models.
        provider_api_keys: dict
            A dict mapping LLM providers to their API keys.
        default_config: dict
            A dict containing the default agent configuration,
            including default LLM provider, model code, etc.
        supported_personalities: list[str] | tuple[str]
            A list or tuple of supported chatbot personalities.
            Each personality string should have matching system prompt
            templates in the directory path AppConfig.CHATBOT_CONTEXT_DIR
        """
        # Check invariant: supported personalities must be list or tuple
        if (
            not isinstance(supported_personalities, (list, tuple))
            or not supported_personalities
        ):
            raise ValueError(
                "Didn't receive a valid value for "
                "`supported_personalities`."
            )
        self.supported_personalities = supported_personalities

        # Set default agent personality from valid options
        default_personality = default_config.get("personality")
        if (
            default_personality is None
            or default_personality not in self.supported_personalities
        ):
            default_personality = self.supported_personalities[0]
        self.default_personality = default_personality

        # Set memory compacint limit
        self.compacting_msg_limit = int(
            default_config.get("compacting message limit", 30)
        )

        # Check invariant: at least one model loaded from app config
        if not isinstance(available_models, dict) or len(available_models) == 0:
            raise ValueError(
                f"Value for `available_models` is invalid: {available_models}"
            )

        # Build available models list
        models = {}
        for provider, details in available_models.items():
            base_url = details["urls"]["api"]
            api_key = provider_api_keys[provider]
            models.update({
                model: {
                    "base url": base_url,
                    "api key": api_key
                } for model in details.get("available models", [])
            })

        # Init parent classes
        ChatCompletionsBaseAgent.__init__(
            self,
            models=models,
            default_model=default_config.get("model"),
            max_recursive_tool_calls=default_config.get("max recursive tool calls"),
            max_completion_tokens=default_config.get("max completion tokens")
        )
        ChatbotContextHelper.__init__(self)

        # Init messages list
        self.messages: list[dict] = []

        # Init personality
        self.set_personality()

    def set_personality(
            self,
            personality: str | None = None
    ) -> None:
        """
        Set the chatbot's instructions based on its selected personality.

        Can be used at any point in the agent's lifecycle to set/update
        its desired behavior towards the user.

        Parameters
        ----------
        personality: str | None
            The personality to use.
            If not provided, defaults to the default personality saved to
            the self.

        Returns
        -------
        None
            Updates the first message in the messages list containing the
            agent's system prompt.

        Notes
        -----
        - Under the current logic, compacting instructions are also dependant
        on the chatbot's selected personality. These are also updated on
        personality changes.
        """
        # Check early conditions
        if (
            personality is None
            or personality not in self.supported_personalities
        ):
            personality = self.default_personality
        if (
            hasattr(self, 'current_personality')
            and self.current_personality == personality
        ):
            return

        # Set agent instructions
        agent_instructions = ChatbotContextHelper.get_agent_instructions(
            self, personality
        )
        instructions_message = {
            "role": "system",
            "content": agent_instructions
        }

        if len(self.messages) > 0:  # Update running agent
            self.messages[0] = instructions_message
        else:  # On agent init
            self.messages.append(instructions_message)

        # Update/set compacting instructions
        self.compacting_instructions = ChatbotContextHelper.get_compacting_instructions(
            self, personality
        )

        # Save current personality
        self.current_personality = personality

    def __call__(
            self,
            user_message: str,
            tools: dict = tool_schema
    ) -> str:
        """
        Call agent with a user message.

        Parameters
        ----------
        user_message: str
            ...
        tools: dict
            Dictionary containing the tools available for the LLM.

        Returns
        -------
        str
            The agent's response.

        Examples
        --------
        >>> agent = ChatbotAssistant(...)
        >>> response = agent(user_message="Hi there.")
        >>> print(response)
            "General Kenobi."
        """
        # Check recursive tool call limit prompt is saved to self
        if not hasattr(self, "rec_tool_call_lim_prompt"):
            self.rec_tool_call_lim_prompt = ChatbotContextHelper.get_rec_tool_lim_prompt(
                self
            )

        # Append user message to list
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # Get new messages resulting from LLM API call
        new_messages = ChatCompletionsBaseAgent.llm_api_call(
            self,
            messages=self.messages,
            tools=tools,
            tool_limit_reached_prompt=self.rec_tool_call_lim_prompt
        )

        # Update messages and save LLM's response
        llm_response = new_messages[-1]["content"]
        self.messages.extend(new_messages)

        # Run compacting check
        self._compact_messages()

        return llm_response

    def _compact_messages(self) -> None:
        """
        Compact the messages list in the current conversation.

        Notes
        -----
        - Makes an LLM API call to summarize the key details in the conversation
        depending on its selected personality.
        - Makes a transcript of the current messages list as a single string,
        then sends as a user message along with the compacting instruction.
        - Updates the chatbot's messages list attribute.
        Keeps two messages in the messages list:
            1. The chatbot instructions
            2. The conversation summary so far
        """
        # Skip if memory update is not needed
        if len(self.messages) <= self.compacting_msg_limit:
            return

        logger.debug(
            f"Messages list length is at limit of {len(self.messages)}. "
            "Compacting..."
        )

        # Check if long-term memory exists so it's included on context
        long_term_memory = (
            self.long_term_memory
            if hasattr(self, "long_term_memory")
            and isinstance(self.long_term_memory, str)
            and len(self.long_term_memory) > 0
            else None
        )

        # Get user prompt and format
        conversation_history = self.messages[1:]  # exclude chatbot instructions
        compacting_user_prompt = ChatbotContextHelper.get_compacting_user_prompt(
            self,
            recent_conversation=conversation_history,
            long_term_memory=long_term_memory
        )

        # Make API call
        messages = [
            {"role": "system", "content": self.compacting_instructions},
            {"role": "user", "content": compacting_user_prompt}
        ]
        conversation_summary = ChatCompletionsBaseAgent.llm_api_call(
            self,
            messages=messages
        )

        # Update messages list
        summary_message = ChatbotContextHelper.get_conversation_summary_prompt(
            self, conversation_summary
        )
        self.messages[1] = {
            "role": "system",
            "content": summary_message
        }
        self.messages = self.messages[: 2]  # remove short-term messages from list
        self.long_term_memory = conversation_summary

    def reset_memory(self) -> None:
        """
        Reset all of the agent's memory, both long and short term.

        Only keeps system instructions based on personality.
        """
        self.messages = self.messages[:1]
