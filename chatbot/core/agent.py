from loguru import logger
from openai import OpenAI

from config import AppConfig
from chatbot.core.context import ChatbotContextHelper


class ChatbotAgent(ChatbotContextHelper):
    """
    Class for chatbot agent that can be used to generate responses based on user input and context.

    Inherits from ChatbotContextHelper for managing the chatbot's context.
    """

    def __init__(
            self,
            available_models: dict = AppConfig.AVAILABLE_MODELS,
            models_api_keys: dict = AppConfig.MODELS_API_KEYS,
            default_chatbot_config: dict = AppConfig.DEFAULT_CONFIG,
            supported_chatbot_personalities: list | tuple = AppConfig.SUPPORTED_CHATBOT_PERSONALITIES,
            fallback_turns_for_context_cleanup: int = 10
    ) -> None:
        """
        Initialize the ChatbotAgent.

        Parameters
        ----------
            available_models: dict
                A dictionary of available LLM providers and their corresponding models,
                as defined in the app config.
            models_api_keys: dict
                A dictionary mapping LLM providers to their corresponding API keys,
                as defined in the app config.
            default_chatbot_config: dict
                A dictionary containing the default configuration for the chatbot, including default
                LLM provider, model code, and other settings.
            supported_chatbot_personalities: list
                A list of supported chatbot personalities.
                Each personality string should have matching system prompt templates in the directory
                path AppConfig.CHATBOT_CONTEXT_DIR
            fallback_turns_for_context_cleanup: int
                A fallback value for the number of conversation turns before context cleanup is triggered,
                in case the value provided in the app config is invalid.

        Attributes
        ----------
            models: dict
                A dictionary for conveniently storing connection parameters for the different available
                LLM providers so different Chatbot clients can be created as needed.
                e.g.,
                    {
                        "gemini-flash-2.5": {
                            "base url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                            "api key": {GEMINI_API_KEY from .env file},
                        }, ...
                    }
            supported_chatbot_personalities: list
                A list of supported chatbot personalities, as defined in the app config.
            memory: dict
                A dictionary to store the chatbot's memory, including long-term and short-term memory.
                * Long-term memory: Key facts about the user and and the conversation that should be retained.
                * Short-term memory: A list of the most recent conversation turns, which are eventually either
                saved to long-term memory or discarded during context cleanup.
        """
        # Raise if no models are available based on the app config
        if not isinstance(available_models, dict) or len(available_models) == 0:
            raise ValueError("Didn't receive a valid value for `available_models`.")

        # Create an object for easier lookup of available models
        self.models = {}
        for provider, details in available_models.items():
            base_url = details["base url"]
            api_key = models_api_keys[provider]
            self.models.update({
                model: {
                    "base url": base_url,
                    "api key": api_key
                } for model in details.get("available models", [])
            })

        # Create an object for storing personality-specific API call parameters
        self.supported_chatbot_personalities = supported_chatbot_personalities
        if (
            not isinstance(self.supported_chatbot_personalities, (list, tuple))
            or not self.supported_chatbot_personalities
        ):
            raise ValueError("Didn't receive a valid value for `supported_chatbot_personalities`.")

        # Set default model to show on Chatbot startup
        default_model = default_chatbot_config.get("model")
        if default_model not in self.models:
            default_model = list(self.models.keys())[0]
        self.default_model = default_model

        # Set default personality to show on Chatbot startup
        default_personality = default_chatbot_config.get("personality")
        if default_personality not in self.supported_chatbot_personalities:
            default_personality = self.supported_chatbot_personalities[0]
        self.default_personality = default_personality

        # Set max completion tokens: if not valid, no limit is set
        max_completion_tokens = default_chatbot_config.get("max completion tokens")
        self.max_completion_tokens = (
                max_completion_tokens if isinstance(max_completion_tokens, int)
                and max_completion_tokens > 0 else None
        )

        turns_for_context_cleanup = default_chatbot_config.get("turns before context cleanup")
        if not isinstance(turns_for_context_cleanup, int) or turns_for_context_cleanup <= 0:
            logger.error(
                f"Invalid value for 'turns before context cleanup': {turns_for_context_cleanup}. "
                f"Defaulting to {fallback_turns_for_context_cleanup}."
            )
            turns_for_context_cleanup = fallback_turns_for_context_cleanup
        self.turns = {
            "context cleanup limit": turns_for_context_cleanup,
            "total": 0
        }

        # Init empty memory
        self.memory = {
            "long term": None,
            "short term": []
        }

        # Initialize context helper
        super().__init__()

    def set_client(
            self,
            model_code: str | None = None,
    ) -> None:
        """
        Set and store an OpenAI client and model code to use.

        This model may be called at any time to update the chatbot's client parameters.

        Parameters
        ----------
            model_code: str | None
                The code of the LLM model for which to create the client.
                If not provided, defaults to the default model saved to the self.

        Returns
        -------
            None
                Sets the self.client attribute to an instance of the OpenAI client.
        """
        if not model_code:
            model_code = self.default_model
        self.model_code = model_code
        model_data = self.models[model_code]
        self.client = OpenAI(
            base_url=model_data["base url"],
            api_key=model_data["api key"]
        )

    def set_personality(
            self,
            personality: str | None = None
    ) -> None:
        """
        Set the chatbot's system prompts to the self, which are determined by its specified personality.

        This method may be called during initialization or at any other time to update
        the chatbot's prompts based on a new input.

        Parameters
        ----------
            personality: str | None
                The personality for which to set the chatbot's system prompts.
                If not provided, defaults to the default personality saved to the self.

        Returns
        -------
            None
                Sets the self.prompts attribute to a dictionary containing the system prompts for the chatbot,
                which are determined by the specified personality.
        """
        if personality not in self.supported_chatbot_personalities:
            personality = self.default_personality

        self.prompts = {
            "chatbot system": self.get_chatbot_system_prompt(personality),
            "memory manager system": self.get_memory_manager_system_prompt(personality),
        }

    def llm_api_call(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        debug: bool = True
    ) -> str | None:
        """
        Call the LLM API and return the generated response.

        Includes special considerations for certain OpenAI models for which parameter
        names have changed.

        Parameters
        ----------
            user_prompt: str
                The user's input prompt to the chatbot.
            system_prompt: Optional[str]
                An optional system prompt to provide additional context or instructions to the LLM.
            tools: Optional[list[dict]]
                An optional list of tools to provide to the LLM for enhanced capabilities.
            debug: bool
                Whether to print debug information about the API call. Defaults to True.

        Returns
        -------
            The generated response from the LLM as a string, or None if the API call fails.
        """
        if not (isinstance(user_prompt, str) and len(user_prompt) > 0):
            raise ValueError("'user_prompt' must be a non-empty string.")

        # Build messages body
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        params = {
            "model": self.model_code,
            "messages": messages
        }
        if hasattr(self, "max_completion_tokens") and self.max_completion_tokens:
            if self.model_code in [
                "gpt-5-mini",
                "gpt-5-nano"
            ]:
                param_name = "max_completion_tokens"
            else:
                param_name = "max_tokens"
            params[param_name] = self.max_completion_tokens
        if isinstance(tools, list) and len(tools) > 0:
            params["tools"] = tools

        # Return API response
        response = self.client.chat.completions.create(**params)
        response_content = response.choices[0].message.content if response.choices else None
        if debug:
            logger.debug(f"LLM API Call - Params: {params}")
            logger.debug(f"LLM API Call - Response: {response_content}")

        return response_content

    def _update_memory(self, debug: bool = True) -> None:
        """
        Update the chatbot's memory based on recent conversation history and long-term memory.

        Cleanup is performed by making an API call which looks to summarize the recent conversation
        and extract any key details to be saved to long-term memory, while discarding the rest.

        Parameters
        ----------
            debug: bool
                Whether to print debug information about the memory update process. Defaults to True.

        Returns
        -------
            None
                Updates the chatbot's memory to the self.memory attribute.
        """
        # skip if memory update is not needed
        if self.turns["total"] % self.turns["context cleanup limit"] != 0:
            return

        # recalculate long-term memory
        new_long_term_memory = self.llm_api_call(
            user_prompt=self.get_memory_manager_user_prompt(
                recent_conversation=self.memory["short term"],
                long_term_memory=self.memory["long term"]
            ),
            system_prompt=self.prompts.get("memory manager system"),
            debug=debug
        )

        # update memory to self
        self.memory["long term"] = new_long_term_memory
        self.memory["short term"] = []

    def chatbot_call(
            self,
            user_query: str,
            debug: bool = True
    ) -> str | None:
        """
        Make a chatbot API call with included context management.

        Parameters
        ----------
            user_query: str
                The user's input query to the chatbot.
            debug: bool
                Whether to print debug information about the chatbot call process. Defaults to True.

        Returns
        -------
            str | None
                The generated response from the chatbot as a string, or None if the API call fails.
        """
        # Memory management
        self._update_memory(debug=debug)

        # Chatbot API call
        system_prompt = self.prompts.get("chatbot system")
        user_prompt = self.get_chatbot_user_prompt(
            user_query=user_query,
            long_term_memory=self.memory["long term"],
            short_term_memory=self.memory["short term"]
        )
        llm_response = self.llm_api_call(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            debug=debug
        )

        # Update memory and turns
        self.memory["short term"].append({
            "user": user_query,
            "chatbot": llm_response
        })
        self.turns["total"] += 1

        return llm_response

    def reset_memory(self) -> None:
        """
        Reset the chatbot's memory, clearing both long-term and short-term memory.

        This can be useful for starting a new conversation or clearing any accumulated context.

        Returns
        -------
            None
                Resets the chatbot's memory to an empty state.
        """
        self.memory = {
            "long term": None,
            "short term": []
        }
