"""
Defines developer-defined configurations to be used in the app.

Notes
-----
    TODO: Add support for self-hosted models, like Ollama and Qwen.

    TODO: Add a parameter for controlling the logging level across the app as
    an env var or config setting.
"""
from config.app_config import AppConfig

__all__ = [
    "AppConfig"
]
