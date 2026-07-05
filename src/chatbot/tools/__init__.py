"""Entrypoint for the tools module for the agent."""
from src.chatbot.tools.builder import build_tools
from src.chatbot.tools.definitions import (
    evaluate_math_expression,
    get_current_date,
    perform_web_search
)
from src.chatbot.tools.web_search import web_search_tool_available

# Build list of available tools
tools = [
    evaluate_math_expression,
    get_current_date,
]

# Special case: add web search only if available
if web_search_tool_available():
    tools.append(perform_web_search)

# Build registry and schema
tool_registry, tool_schema = build_tools(tools)

__all__ = [
    "tool_registry",
    "tool_schema"
]
