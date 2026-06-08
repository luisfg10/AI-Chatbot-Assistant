"""
Module for defining the chatbot's context and logic.

Notes
-----
    TODO: Optimize prompt management by loading the templates into memory once and
    then retrieving and formatting the relevant sections as needed, as opposed
    to loading them each time a prompt is needed.

    TODO: Support dynamic model selection for each API call. The user should see a dropdown
    list of available models for certain providers, and that model should be used to determine
    what API endpoint to call, similar to GitHub Copilot's model selection list.

"""
from .core.agent import ChatbotAgent

__all__ = ["ChatbotAgent"]
