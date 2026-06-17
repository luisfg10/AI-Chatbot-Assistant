"""Define default settings for the application."""
# Built-ins
import json
import os
import sys

from dotenv import load_dotenv
from loguru import logger


# Class for managing environment variables
class AppConfig:
    """
    Utility class for managing the app's configurations and settings.

    Notes
    -----
        - Use llm_config.json to define available LLM providers, their models and chat requests
        settings. Base URLs should be compatible with OpenAI's Python SDK.
        - Use the .env file to set the API keys for LLM providers.
        - Available models for the UI are resolved as the set of available models for which the provider
        has a set API key defined in the .env file. Doesn't yet support self-hosted models.
    """

    # Default Directories (should end with "/")
    CHATBOT_CONTEXT_DIR: str = "src/chatbot/context/"
    DOTENV_FILE_PATH: str = "config/.env"
    LLM_CONFIG_PATH: str = "config/llm_config.json"

    # Load reference json for llm config
    if os.path.exists(LLM_CONFIG_PATH):
        with open(LLM_CONFIG_PATH) as f:
            LLM_CONFIG: dict = json.load(f)
    else:
        raise FileNotFoundError(
            "Application cannot start: "
            f"LLM config file not found at '{LLM_CONFIG_PATH}'"
        )
    DEFAULT_CONFIG: dict = LLM_CONFIG.get("default config", {})
    SUPPORTED_CHATBOT_PERSONALITIES: list = LLM_CONFIG.get("supported chatbot personalities", [])

    # ------------------------------------------------------------------
    # Environment Variables

    # Load .env file
    load_dotenv(
        os.path.join(os.getcwd(), f"{DOTENV_FILE_PATH}"),
        override=True
    )

    # Resolve available models and providers
    AVAILABLE_MODELS: dict = {
        provider: details
        for provider, details in LLM_CONFIG.get("providers", {}).items()
        if str(os.getenv(f"{provider.upper()}_API_KEY")).strip() != ""
    }
    PROVIDER_API_KEYS: dict = {
        provider: os.getenv(f"{provider.upper()}_API_KEY")
        for provider in AVAILABLE_MODELS.keys()
    }

    # Determine logging level
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    if LOG_LEVEL not in logger._core.levels:
        raise ValueError(
            f"Invalid LOG_LEVEL provided: '{LOG_LEVEL}'. "
            f"Valid options: {list(logger._core.levels.keys())}"
        )
    if LOG_LEVEL != "DEBUG":
        logger.remove()
        logger.add(sys.stderr, level=LOG_LEVEL)

    # Log initialized config
    logger.info(
        f"Initialized AppConfig with available providers: {list(AVAILABLE_MODELS.keys())} "
        f"and default config: {DEFAULT_CONFIG}"
    )
