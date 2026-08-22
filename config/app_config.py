"""Define default settings for the application."""

import json
import os
import sys

from dotenv import load_dotenv
from loguru import logger


class AppConfig:
    """Utility class for managing app settings."""

    # Default Directories (should end with "/")
    CHATBOT_CONTEXT_DIR: str = "src/chatbot/context/"
    DOTENV_FILE_PATH: str = "config/.env"
    LLM_CONFIG_PATH: str = "config/llm_config.json"

    # Load llm_config file
    if os.path.exists(LLM_CONFIG_PATH):
        with open(LLM_CONFIG_PATH) as f:
            LLM_CONFIG: dict = json.load(f)
    else:
        raise FileNotFoundError(
            "Application cannot start: "
            f"LLM config file not found at '{LLM_CONFIG_PATH}'"
        )
    DEFAULT_CONFIG: dict = LLM_CONFIG.get("default config", {})
    SUPPORTED_CHATBOT_PERSONALITIES: list = LLM_CONFIG.get(
        "supported chatbot personalities", []
    )

    # ------------------------------------------------------------------
    # Environment Variables

    # Load .env file from specified path
    load_dotenv(
        os.path.join(os.getcwd(), f"{DOTENV_FILE_PATH}"),
        override=True  # overridde any existing env vars
    )

    # Resolve available LLMs and providers
    AVAILABLE_MODELS: dict = {
        provider: details
        for provider, details in LLM_CONFIG.get("providers", {}).items()
        if str(os.getenv(f"{provider.upper()}_API_KEY")).strip() != ""
    }
    PROVIDER_API_KEYS: dict = {
        provider: os.getenv(f"{provider.upper()}_API_KEY")
        for provider in AVAILABLE_MODELS.keys()
    }

    # Tavily credentials (web search tool)
    try:
        tavily_url = LLM_CONFIG["tools"]["web search"]["tavily"]
    except KeyError:
        tavily_url = None
    TAVILY_CONFIG: dict = {
        "url": tavily_url,
        "api key": os.getenv("TAVILY_API_KEY")
    }

    # Resolve logging level
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    if LOG_LEVEL not in logger._core.levels:  # Check invalid value
        logger.error(
            f"Invalid LOG_LEVEL provided: '{LOG_LEVEL}'. "
            f"Valid options: {list(logger._core.levels.keys())}.\n"
            "Continuing with default LOG_LEVEL."
        )
    if LOG_LEVEL != "DEBUG":
        logger.remove()
        logger.add(sys.stderr, level=LOG_LEVEL)

    # Log initialized config
    logger.info(
        f"Initialized AppConfig with "
        f"available LLM providers: {list(AVAILABLE_MODELS.keys())} "
        f"and default config: {DEFAULT_CONFIG}"
    )
