"""
Entry point for the application's frontend logic.

Notes
-----
    * When modiying parameters that impact the calls to the LLM, the backend call to the API
    must be set to await confirmation of the change before allowing the next call to be made,
    in order to prevent race conditions.

"""
from .ui import build_chatbot_ui

__all__ = ["build_chatbot_ui"]
