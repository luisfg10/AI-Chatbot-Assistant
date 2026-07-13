from loguru import logger

from config import AppConfig
from src.chatbot.core.base_agent import ChatCompletionsBaseAgent
from src.chatbot.core.context import ChatbotContextHelper
from src.chatbot.tools import tool_schema


class ChatbotAssistant(ChatCompletionsBaseAgent, ChatbotContextHelper):
    """
    Class for chatbot that interacts with a user via a chat interface.

    Inherits from ChatCompletionsBaseAgent for LLM API interactions and from
    ChatbotContextHelper for managing the chatbot's context.
    """

    def __init__(
            self,
            available_models: dict = AppConfig.AVAILABLE_MODELS,
            provider_api_keys: dict = AppConfig.PROVIDER_API_KEYS,
            default_config: dict = AppConfig.DEFAULT_CONFIG,
            supported_personalities: list | tuple = AppConfig.SUPPORTED_CHATBOT_PERSONALITIES
    ) -> None:
        """
        Initialize the ChatbotAssistant.

        Parameters
        ----------
            available_models: dict
                A dictionary of available LLM providers and their corresponding
                models, as defined in the app config.
            provider_api_keys: dict
                A dictionary mapping LLM providers to their corresponding API keys,
                as defined in the app config.
            default_config: dict
                A dictionary containing the default configuration for the chatbot,
                including default LLM provider, model code, and other settings.
            supported_personalities: list
                A list of supported chatbot personalities.
                Each personality string should have matching system prompt
                templates in the directory path AppConfig.CHATBOT_CONTEXT_DIR
        """
        # Create an object for storing personality-specific API call parameters
        self.supported_personalities = supported_personalities
        if (
            not isinstance(self.supported_personalities, (list, tuple))
            or not self.supported_personalities
        ):  # Check invariant: supported personalities must be list or tuple
            raise ValueError(
                "Didn't receive a valid value for "
                "`supported_personalities`."
            )
        # Set default personality to show on Chatbot startup
        default_personality = default_config.get("personality")
        if default_personality not in self.supported_personalities:
            default_personality = self.supported_personalities[0]
        self.default_personality = default_personality

        # Set total messages in messages list before memory compacting
        self.compacting_msg_limit = int(
            default_config.get("compacting message limit", 30)
        )

        # Check invariant: at least one model loaded from app config
        if not isinstance(available_models, dict) or len(available_models) == 0:
            raise ValueError(
                f"Value for `available_models` is invalid: {available_models}"
            )
        # Build the list of available models
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

        # Initialize parent classes
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

        # Initialize personality
        self.set_personality()

    def set_personality(
            self,
            personality: str | None = None
    ) -> None:
        """
        Set the chatbot's instructions depending on its selected personality.

        This method may be called during initialization or at any other time
        to update the chatbot's prompts based on a new input.
        Modifies the first message in the messages list to the self.

        Parameters
        ----------
            personality: str | None
                The personality for which to set the chatbot's system prompts.
                If not provided, defaults to the default personality saved to
                the self.

        Returns
        -------
            None
                Updates the first message in the messages list containing the
                chatbot's isntructions.
                In the case of the instructions for compacting, saves the
                instructions to use to the self.
        """
        if personality not in self.supported_personalities:
            personality = self.default_personality

        # Chatbot instructions
        chatbot_instructions = ChatbotContextHelper.get_chatbot_instructions(
            self, personality
        )
        instructions_message = {
            "role": "system",
            "content": chatbot_instructions
        }

        if len(self.messages) > 0:
            self.messages[0] = instructions_message
        else:
            self.messages.append(instructions_message)

        # Compacting instructions
        self.compacting_instructions = ChatbotContextHelper.get_compacting_instructions(
            self, personality
        )

    def chatbot_call(
            self,
            user_query: str,
            tools: dict = tool_schema
    ) -> str | None:
        """
        Make a chatbot API call with included context management.

        Parameters
        ----------
            user_query: str
                The user's input query to the chatbot.
            tools: dict
                Dictionary containing the tools the LLM can call.

        Returns
        -------
            str | None
                The generated response from the chatbot as a string,
                or None if the API call fails.
        """
        # Check that recursive tool call limit prompt is saved to self
        if not hasattr(self, "rec_tool_call_lim_prompt"):
            self.rec_tool_call_lim_prompt = self.get_rec_tool_lim_prompt()

        # Append user query to messages
        self.messages.append({
            "role": "user",
            "content": user_query
        })

        new_messages = ChatCompletionsBaseAgent.llm_api_call(
            self,
            messages=self.messages,
            tools=tools,
            tool_limit_reached_prompt=self.rec_tool_call_lim_prompt
        )
        llm_response = None

        # Update messages and save LLM's response
        llm_response = new_messages[-1]["content"]
        self.messages.extend(new_messages)

        # Evaluate compacting
        self._compact_messages()

        return llm_response

    def _compact_messages(self) -> None:
        """
        Run compacting to summarize the key details in the current conversation.

        Makes a transcript of the current messages list as a single string,
        then sends as a user message along with the compacting instruction.

        Returns
        -------
            None
                Updates the chatbot's messages list attribute.
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

        # Check if long-term memory exists
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
        Reset the chatbot's memory, clearing both long-term and short-term memory.

        This can be useful for starting a new conversation or clearing any
        accumulated context.

        Returns
        -------
            None
                Resets the chatbot's memory to an empty state, outside of its
                instructions based on personality.
        """
        self.messages = self.messages[:1]
