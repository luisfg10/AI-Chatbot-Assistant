"""
Module for defining the chatbot's context and logic.

Notes
-----
    TODO: Change AI endpoint from OpenAI chat completions to responses

    TODO: Add web search tool and connect to frontend, possibly using an external
    API like SerpAPI or Tavily.

    TODO: Add support for a wider range of personalities, and allow users to define
    their custom personalities, which should also be persisted across sessions.

    TODO: Switch to more advanced agent memory and compacting techniques.

    TODO: Add a max number of tool executions before an answer in order to avoid
    infinite loops.
"""
from .core.agent import ChatbotAgent

__all__ = ["ChatbotAgent"]
