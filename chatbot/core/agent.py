from openai import OpenAI

from config import AppConfig
from chatbot.utils.context import ChatbotContextHelper


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
            create_client: bool
                Whether to create the OpenAI client during initialization. Defaults to True.
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

    def call_llm(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None
    ) -> str | None:
        """
        Call the LLM API and return the generated response.

        Parameters
        ----------
            user_prompt: str
                The user's input prompt to the chatbot.
            model: str
                The specific LLM model code to use for generating the response.
            system_prompt: Optional[str]
                An optional system prompt to provide additional context or instructions to the LLM.
            max_tokens: Optional[int]
                The maximum number of tokens to generate in the response.
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
        if isinstance(max_tokens, int) and max_tokens > 0:
            params["max_completion_tokens"] = max_tokens
        if isinstance(tools, list) and len(tools) > 0:
            params["tools"] = tools

        # Return API response
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content if response.choices else None
