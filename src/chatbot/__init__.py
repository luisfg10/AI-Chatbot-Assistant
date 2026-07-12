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

    TODO: Allow for tool-calling loops and add a max number of tool executions before
    an answer is provided. Currently only one tool call turn is allowed before returning
    a response.

    TODO: Use proper type hints for OpenAI's return objects on functions and
    class methods.
"""
from .core.agent import ChatbotAssistant

__all__ = ["ChatbotAssistant"]
