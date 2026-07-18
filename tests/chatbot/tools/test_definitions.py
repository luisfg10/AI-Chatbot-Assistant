import re

import pytest

from src.chatbot.tools.definitions import (
    get_current_date,
)


class TestGetCurrentDate:
    """
    Test the get_current_date function non-exhaustively.

    Notes
    -----
        * Pending test for actually returning today's date, seems
        like an overkill.
    """

    @pytest.fixture
    def result(self) -> str:
        """Use subject function to get today's date."""
        return get_current_date()

    @pytest.fixture
    def parsed(self, result: str) -> tuple:
        """Parse the function's output into its constituents."""
        match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", result)
        assert match is not None, f"'{result}' is not in YYYY-MM-DD format"
        return int(match[1]), int(match[2]), int(match[3])

    def test_matches_expected_format(self, result: str) -> None:
        """
        Check that function output has expected format.

        Checks four digits, followed by a hyphen, followed by two digits,
        and so on, effectively validating YYYY-MM-DD.
        """
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", result)

    def test_year_is_plausible(self, parsed: tuple) -> None:
        """Test output year makes sense."""
        year = parsed[0]
        assert 2026 <= year <= 2050

    def test_month_is_valid(self, parsed: tuple) -> None:
        """Test output month is between 1 and 12."""
        month = parsed[1]
        assert 1 <= month <= 12

    def test_day_is_valid(self, parsed: tuple) -> None:
        """Test output day is between 1 and 31."""
        day = parsed[2]
        assert 1 <= day <= 31


class EvaluateMathExpression:
    """Test the evaluate_math_expression function."""


class PerformWebSearch:
    """Test the perform_web_search function."""
