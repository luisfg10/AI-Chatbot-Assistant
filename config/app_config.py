"""Define default settings for the application."""
# Built-ins
import os
import json

# Third-party
from loguru import logger
from dotenv import load_dotenv


# Class for managing environment variables
class AppConfig:
    """Utility class for managing the app's configurations and settings."""

    # Default Directories (should end with "/")
    CHATBOT_CONTEXT_DIR: str = "chatbot/context/"
    DOTENV_FILE_PATH: str = "config/.env"
    LLM_CONFIG_PATH: str = "config/llm_config.json"

    # Allowed values for variables
    SUPPORTED_CHATBOT_PERSONALITIES: list | tuple = (
        "friendly",
        "professional"
    )

    # Load reference json for LLM providers and models
    if os.path.exists(LLM_CONFIG_PATH):
        with open(LLM_CONFIG_PATH) as f:
            LLM_CONFIG = json.load(f)
    else:
        raise FileNotFoundError(
            "Application cannot start: "
            f"LLM config file not found at '{LLM_CONFIG_PATH}'"
        )

    # ------------------------------------------------------------------
    # Environment Variables

    # Load .env file
    load_dotenv(
        os.path.join(os.getcwd(), f"{DOTENV_FILE_PATH}"),
        override=True
    )

    # Set LLM provider
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER")
    if not LLM_PROVIDER:
        LLM_PROVIDER = LLM_CONFIG.get("default provider")
        if not LLM_PROVIDER:
            raise ValueError(
                "LLM_PROVIDER not set in environment variables and no default "
                "provider specified in LLM config."
            )
        logger.warning(
            "LLM_PROVIDER not set in environment variables. "
            f"Defaulting to '{LLM_PROVIDER}'"
        )
    if LLM_PROVIDER not in LLM_CONFIG.get("available providers", {}):
        raise ValueError(
            f"LLM_PROVIDER '{LLM_PROVIDER}' is not in the list of available "
            "providers specified in LLM config.\n"
            f"Available providers: {list(LLM_CONFIG.get('available providers', {}).keys())}"
        )

    # Set LLM
    LLM_CODE: str = os.getenv("LLM_CODE")
    if not LLM_CODE:
        LLM_CODE = LLM_CONFIG.get(
            "available providers"
        ).get(LLM_PROVIDER).get("default model")
        if not LLM_CODE:
            raise ValueError(
                "LLM_CODE not set in environment variables and no default "
                f"model specified for provider '{LLM_PROVIDER}' in LLM config."
            )
        logger.warning(
            "LLM_CODE not set in environment variables. "
            f"Defaulting to '{LLM_CODE}' for provider '{LLM_PROVIDER}'"
        )

    # Set Temperature
    LLM_TEMPERATURE: float = float(
        os.getenv("LLM_TEMPERATURE"),
        LLM_CONFIG.get("default temperature", 0.7)
    )

    # Set API Key, if required by provider
    LLM_API_KEY: str = os.getenv("LLM_API_KEY")
