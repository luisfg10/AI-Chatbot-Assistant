from openai import OpenAI

from config import AppConfig
from chatbot.core.context import ChatbotContextHelper


class ChatbotAgent(ChatbotContextHelper):
    """
    Class for chatbot agent that can be used to generate responses based on user input and context.

    Inherits from ChatbotContextHelper for managing the chatbot's context.

    For simplicity, even though this class can be used to connect to multiple providers, only those
    who provide compatible endpoints to OpenAI's wrapper library are supported.
    """

    # ------------------------------------------------------------------
    # Class Attributes

    LLM_PROVIDER_URL_MAPPINGS: dict[str, str | None] = {
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "openai": None  # defaults to OpenAI's default base URL
    }

    def __init__(
            self,
            llm_provider: str = AppConfig.LLM_PROVIDER,
            llm_code: str = AppConfig.LLM_CODE,
            llm_api_key: str | None = AppConfig.LLM_API_KEY,
            max_completion_tokens: int | None = AppConfig.LLM_MAX_COMPLETION_TOKENS,
            supported_llm_personalities: list | tuple = AppConfig.SUPPORTED_CHATBOT_PERSONALITIES,
            turns_for_context_cleanup: int = AppConfig.LLM_TURNS_BEFORE_CONTEXT_CLEANUP,
            create_client: bool = True
    ) -> None:
        """
        Initialize the ChatbotAgent.

        Parameters
        ----------
            llm_provider: Optional[str]
                The LLM provider to use (e.g., "gemini", "openai").
                If not provided, it will be read from environment variables or defaulted.
            llm_code: Optional[str]
                The specific LLM model code to use (e.g., "gemini-2.0-pro", "gpt-4").
                If not provided, it will be read from environment variables or defaulted.
            llm_api_key: Optional[str]
                The API key for the LLM provider. If not provided, it will be read from environment variables.
            max_completion_tokens: int | None = AppConfig.LLM_MAX_COMPLETION_TOKENS,
                The maximum number of tokens for LLM completions.
                Defaults to the value from AppConfig.
            supported_llm_personalities: list | tuple = AppConfig.SUPPORTED_CHATBOT_PERSONALITIES,
                The list of supported LLM personalities.
                Defaults to the value from AppConfig.
            turns_for_context_cleanup: int = AppConfig.LLM_TURNS_BEFORE_CONTEXT_CLEANUP,
                The number of conversation turns after which to trigger context cleanup.
                Defaults to the value from AppConfig.
            create_client: bool
                Whether to create the OpenAI client during initialization. Defaults to True.

        Attributes
        ----------
            ...
            memory: dict
                A dictionary to store the chatbot's memory, including long-term and short-term memory.
                * Long-term memory: Key facts about the user and and the conversation that should be retained.
                * Short-term memory: A list of the most recent conversation turns, which are eventually either saved
                to long-term memory or discarded during context cleanup.
        """
        if not all(
            isinstance(param, str) and len(param) > 0
            for param in [llm_provider, llm_code]
        ):
            raise ValueError(
                "'llm_provider' and 'llm_code' both must be non-empty strings."
            )
        if llm_provider not in self.__class__.LLM_PROVIDER_URL_MAPPINGS:
            raise ValueError(
                f"'llm_provider' not recognized for API URL mappings. "
                f"Available providers: {list(self.__class__.LLM_PROVIDER_URL_MAPPINGS.keys())}"
            )
        self.llm_code = llm_code
        self.supported_llm_personalities = (
            supported_llm_personalities
            if isinstance(supported_llm_personalities, (list, tuple))
            and len(supported_llm_personalities) > 0
            else None
        )
        self.max_completion_tokens = (
            max_completion_tokens
            if isinstance(max_completion_tokens, int) and max_completion_tokens > 0
            else None
        )
        self.turns_for_context_cleanup = turns_for_context_cleanup

        # Create Client if requested, otherwise save parameters for later creation
        if create_client:
            base_url = self.__class__.LLM_PROVIDER_URL_MAPPINGS.get(llm_provider)
            self.client = self._create_client(
                base_url=base_url,
                api_key=llm_api_key
            )
        else:
            self.llm_provider = llm_provider
            self.llm_api_key = llm_api_key

        # Init empty memory
        self.memory = {
            "long term": None,
            "short term": []
        }

        # Initialize context helper
        super().__init__()

    @staticmethod
    def _create_client(
            base_url: str,
            api_key: str
    ) -> OpenAI:
        """
        Create and return an OpenAI client based on the specified LLM provider.

        Parameters
        ----------
            base_url: str
                The base URL for the LLM API, which may vary based on the LLM provider.
            api_key: str
                The API key for authenticating with the base URL.

        Returns
        -------
            An instance of the OpenAI client configured for the specified provider.
        """
        return OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def set_prompt_templates(self, personality: str) -> None:
        """
        Set the chatbot's system prompts in the self, which are determined by its specified personality.

        This method may be called during initialization or at any other time to update
        the chatbot's prompts based on a new input.
        """
        if hasattr(self, "supported_llm_personalities") and self.supported_llm_personalities:
            if personality not in self.supported_llm_personalities:
                raise ValueError(
                    f"Personality '{personality}' is not in the list of supported "
                    f"personalities: {self.supported_llm_personalities}"
                )

        self.prompts = {
            "chatbot system": self.get_chatbot_system_prompt(personality),
            "memory manager system": self.get_memory_manager_system_prompt(personality),
        }

    def llm_api_call(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None
    ) -> str | None:
        """
        Call the LLM API and return the generated response.

        Parameters
        ----------
            user_prompt: str
                The user's input prompt to the chatbot.
            system_prompt: Optional[str]
                An optional system prompt to provide additional context or instructions to the LLM.
            tools: Optional[list[dict]]
                An optional list of tools to provide to the LLM for enhanced capabilities.

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
            "model": self.llm_code,
            "messages": messages
        }
        if hasattr(self, "max_completion_tokens") and self.max_completion_tokens:
            params["max_tokens"] = self.max_completion_tokens
        if isinstance(tools, list) and len(tools) > 0:
            params["tools"] = tools

        # Return API response
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content if response.choices else None

    def _cleanup_context(self) -> None:
        """
        Perform context cleanup based on the number of conversation turns.

        Does not validate the appropriate number of turns for cleanup, this should be done before
        calling this method.
        """
        # TODO: Complete after implementing memory manager prompts and functionality

