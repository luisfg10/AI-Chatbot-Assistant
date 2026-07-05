"""Entrypoint for the tools module for the agent."""
from src.chatbot.tools.builder import build_tools
from src.chatbot.tools.definitions import (
    evaluate_math_expression,
    get_current_date
)

# Build list of available tools
tools = [
    evaluate_math_expression,
    get_current_date
]

# Build registry and schema
tool_registry, tool_schema = build_tools(tools)

__all__ = [
    "tool_registry",
    "tool_schema"
]
