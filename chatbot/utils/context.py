from typing import Optional

from config import AppConfig


class ChatbotContextHelper:
    """Utility code for loading and managing the chatbot's context."""

    def __init__(
            self,
            context_dir: Optional[str] = AppConfig.CHATBOT_CONTEXT_DIR
    ) -> None:
        """Initialize the ChatbotContextHelper."""
        if not isinstance(context_dir, str):
            raise ValueError("'context_dir' must be a string.")
        if not context_dir.endswith("/"):
            context_dir += "/"
        self.context_dir = context_dir
