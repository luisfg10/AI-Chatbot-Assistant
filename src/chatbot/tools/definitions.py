"""
Tool definitions to be used by the AI agent.

Consistent and valid numpy-style docstring documentation
is mandatory so the tool schema builder parses the docstring
correctly, otherwise it will raise a ValueError.

Keep each function's docstring informative and concise to improve
the chances it's called correctly by the LLM. Avoid overly describing
parameters and return values without need.
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
    Certain characters are translated as valid operators,
    like "x" into "*" for multiplication.
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
    Perform a web search for grounding a user's query.

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
