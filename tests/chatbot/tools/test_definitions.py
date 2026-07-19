import re

import pytest

from src.chatbot.tools.definitions import evaluate_math_expression, get_current_date


class TestGetCurrentDate:
    """
    Test the `get_current_date` function.

    Notes
    -----
        TODO: Test the function actually returns today's date.
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


class TestEvaluateMathExpression:
    """
    Test the `evaluate_math_expression` function.

    Notes
    -----
        * By design, this function is an agent-facing wrapper
        of a more thorough implementation. Invalid expressions
        are covered in `test_math.py`.
    """

    @pytest.mark.parametrize("expression, result", [
        ("2 + 2", "4"),
        ("3 * (9 + 2) / 1.264", "26.11"),
        ("2 ** 10 / 3.14", "326.11"),
        ("1+1+1+1+1+1+1+1/34", "7.03"),
        ("1-2+3-4+5-6+7", "4")
    ])
    def test_valid_expressions(
        self,
        expression: str,
        result: str,
        roundoff: int = 2
    ) -> None:
        """
        Test the tool returns the correct result for valid expressions.

        Parameters
        ----------
            ...
            roundoff: int = 2
                The number of decimals to which to round the obtained result,
                so that the test doesn't fail because of rounding errors.
        """
        obtained_result = round(
            float(evaluate_math_expression(expression)), roundoff
        )
        expected_result = round(float(result), roundoff)
        assert obtained_result == expected_result

    @pytest.mark.parametrize("expression, result", [
        ("2 x 2", "4"),
        ("2 X 3", "6"),
        ("5X9", "45"),
        ("7x2", "14"),
        ("1^2", "1"),
        ("5^3x1", "125")
    ])
    def test_convertable_valid_expressions(
        self,
        expression: str,
        result: str,
        roundoff: int = 2
    ) -> None:
        """
        Test expressions that can be converted into valid.

        Characters like "X" and "^", while not directly valid,
        should be interpreted by the function and translated to its
        appropriate, valid character so the expression can be evaluated.

        Parameters
        ----------
            ...
            roundoff: int = 2
                The number of decimals to which to round the obtained result,
                so that the test doesn't fail because of rounding errors.
        """
        obtained_result = round(
            float(evaluate_math_expression(expression)), roundoff
        )
        expected_result = round(float(result), roundoff)
        assert obtained_result == expected_result


class TestPerformWebSearch:
    """Test the perform_web_search function."""
