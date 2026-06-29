"""
Tool definitions to be used by the AI agent.

Keep each function's docstring informative and concise
to improve the chances it's called correctly by the LLM.
Avoid defining parameters and return values unnecessarily.

"""
from datetime import date


def get_current_date() -> str:
    """Get today's date in YYYY-MM-DD."""
    return date.today().isoformat()
