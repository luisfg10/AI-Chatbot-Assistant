"""
Tool definitions to be used by the AI agent.

Consistent and valid numpy-style docstring documentation
is mandatory so the tool schema builder parses the docstring
correctly on runtime, otherwise it will raise a ValueError.

Keep each function's docstring informative and concise to improve
the chances it's called correctly by the LLM.

TODO: Improve web search tool so agent is free to choose whether to
get a simplified response or the entire metadata, maybe even return all
useful metadata by default. Sometimes the summary headline returned by
Tavily doesn't have all of the needed information, like the sources
from which it was pulled.
"""
from datetime import date

from src.chatbot.tools.math import CalculatorError, calculate
from src.chatbot.tools.web_search import TavilyClient


def get_current_date() -> str:
    """Get today's date in YYYY-MM-DD format."""
    return date.today().isoformat()


def evaluate_math_expression(expression: str) -> str:
    """
    Evaluate a math expression and return the result.

    Supports int, float, +, -, *, /, ** and parentheses.
    "X" is treated as a valid multiplication operator.
    No other operators, functions, or variables are allowed.

    Parameters
    ----------
    expression: str
        A math expression as a string, e.g. "3 * (8 + 1) / 2".

    Returns
    -------
    str
        The result value as a string, or error reason.
    """
    try:
        result = calculate(expression)
        return str(result)
    except CalculatorError as e:
        return f"Error: {e}"


def perform_web_search(query: str) -> str:
    """
    Perform a web search from a given query.

    Parameters
    ----------
    query: str
        Text to search. e.g., "Tesla stock price NYSE today"

    Returns
    -------
    str
        The results from the web search.
    """
    return TavilyClient().search(
        query=query,
        simplify_response_for_agent=True
    )
