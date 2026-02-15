from typing import Optional

from openai import OpenAI

from config import AppConfig


class ChatbotAgent:
    """
    Class for chatbot agent that can be used to generate responses based on user input and context.

    For simplicity, even though this class can be used to connect to multiple providers, only those
    who provide compatibility to OpenAI's library are supported.
    """

    # Define class attrivutes
    def __init__(
            self,
            llm_provider: Optional[str] = AppConfig.LLM_PROVIDER,
            llm_code: Optional[str] = AppConfig.LLM_CODE,
            llm_api_key: Optional[str] = None

    ) -> None:
        pass
