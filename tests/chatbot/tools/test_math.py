import random

import pytest

from src.chatbot.tools.math import (
        MAX_BASE_FOR_LARGE_EXPONENT,
        MAX_DECIMAL_PLACES,
        MAX_EXPONENT_FOR_LARGE_BASE_CHECK,
        MAX_EXPONENT_MAGNITUDE,
        MAX_EXPRESSION_LENGTH,
        MAX_OPERATOR_COUNT,
        CalculatorError,
        calculate,
)

# ------------------------------------------------------------------
# Utils for Testing


def build_test_expression(
        pass_expr_len: bool,
        pass_op_count: bool,
        max_expr_len: int = MAX_EXPRESSION_LENGTH,
        max_op_count: int = MAX_OPERATOR_COUNT,
        test_cases: int = 5,
        random_seed: int = 50
) -> list[str]:
    """
    Build a random expression meeting specified char and operator lengths.

    Each term within each generated test expression has the same uniform
    value (see examples below).

    Parameters
    ----------
        pass_expr_len: bool
            Whether or not the expression should meet the
            `max_expression_length` value.
        pass_op_count: bool
            Whether or not the expression should meet the
            `max_operator_count` value.
        max_expr_len: int
            The max characters allowed in an expression in order for
            it to be considered valid.
        max_op_count: int
            The max number of operators allowed in an expression.
            This is taken into account to eliminate possible overlap
            with the previous parameter, but isn't tested here.
        test_cases: int
            The number of tests to generate meeting the criteria.
        random_seed: int
            The seed for random number generation.
            Important for guaranteeing reproducibility of results.

    Returns
    -------
        list[str]
            A list of math expressions in str form meeting the
            required criteria.

    Examples
    --------
    >>> example = TestMath().build_test_expression(
        pass_expr_len=False,
        pass_op_count=True,
        max_expr_len=10,
        max_op_count=5,
        test_cases=3
    )
    >>> print(example)
    [
        '111111+111111+111111',
        '111111111111111+111111111111111',
        '11111+11111+11111+11111'
    ]
    """
    # Seed randomness
    rd = random.Random(random_seed)
    # Determine total operators to use per generated test
    if pass_op_count:
        operator_counts = [
                rd.randint(1, max_op_count)
                for _ in range(1, test_cases + 1)
        ]
    else:
        operator_counts = [
            rd.randint(max_op_count + 1, max_op_count * 2)
            for _ in range(1, test_cases + 1)
        ]

    # Determine total digits to use per test
    # chars = digits per term + (digits per term + 1) * # ops
    # digits per term = (# chars - # ops) / (# ops + 1)
    if pass_expr_len:
        digits_per_term = [
            rd.randint(
                1,
                (max_expr_len - operator_counts[i]) // (operator_counts[i] + 1)
            )
            for i in range(0, test_cases)
        ]
    else:
        digits_per_term = [
            rd.randint(
                (max_expr_len - operator_counts[i]) // (operator_counts[i] + 1) + 1,
                2 * (max_expr_len - operator_counts[i]) // (operator_counts[i] + 1)
            )
            for i in range(0, test_cases)
        ]

    # Build test expressions
    tests = []
    for i in range(0, test_cases):
        term = "1" * digits_per_term[i]
        all_test_terms = [term] * (operator_counts[i] + 1)
        tests.append("+".join(all_test_terms))

    return tests


# ------------------------------------------------------------------
# Testing Class

class TestMath:
    """
    Test class for the math module inside tools.

    Includes more thorough testing of invalid expressions
    and special conditions than in `test_definitions.py`.

    Notes
    -----
        TODO: Tie the values of the constants being enforced to
        pytest fixtures and vary them randomly. This will provide
        more robust tests than using fixed values, even if these are
        the ones being used for the application.

        TODO: Improve testing consistency. The tests
            `test_expr_len_pass`
            `test_max_expr_len_fail`
            `test_max_op_count_fail`
        all have thorough, randomized (and perhaps overengineered)
        tests, while other tests in the stack are simple one-offs.
        Choose one of the two approaches, and apply consistently
        everywhere.
    """

    @pytest.mark.parametrize("expression", [
        "import os",
        "print('Hello World!')",
        "os.system('rm -rf /')",
        "import pandas as pd",
        "x = 1",
        "3 +3 = 6"
    ])
    def test_code_exec_forbidden(self, expression: str) -> None:
        """
        Test that code execution calls raise a CalculatorError.

        Parameters
        ----------
            expression: str
                The expression to evaluate.
        """
        with pytest.raises(CalculatorError):
            calculate(expression)

    @pytest.mark.parametrize("expression", build_test_expression(
        pass_expr_len=True,
        pass_op_count=True,
        test_cases=10
    ))
    def test_expr_len_pass(self, expression: str) -> None:
        """
        Test expressions meeting character and operators limits are let run.

        Constants being directly tested for happy cases:
            - MAX_EXPRESSION_LENGTH
            - MAX_OPERATOR_COUNT

        Also tests the other constants indirectly.
        """
        assert isinstance(
            calculate(expression), (int, float)
        )

    @pytest.mark.parametrize("expression", build_test_expression(
        pass_expr_len=False,
        pass_op_count=True,
        test_cases=10
    ))
    def test_max_expr_len_fail(self, expression: str) -> None:
        """Test `MAX_EXPRESSION_LENGTH` is enforced for invalid expressions."""
        with pytest.raises(
            CalculatorError,
            match="characters long, which exceeds"
        ):
            calculate(expression)

    @pytest.mark.parametrize("expression", build_test_expression(
        pass_expr_len=True,
        pass_op_count=False,
        test_cases=10
    ))
    def test_max_op_count_fail(self, expression: str) -> None:
        """Test `MAX_OPERATOR_COUNT` is enforced for invalid expressions."""
        with pytest.raises(
            CalculatorError,
            match="operators, which exceeds the maximum"
        ):
            calculate(expression)

    def test_max_exponent_magnitude_fail(
            self,
            max_exp_mag: int = MAX_EXPONENT_MAGNITUDE
    ) -> None:
        """
        Test `MAX_EXPONENT_MAGNITUDE` is enforced for invalid expressions.

        Parameters
        ----------
            max_exp_mag: int
                Max value for exponents, independent of their base.

        Notes
        -----
            This test is very simplistic, which may not be consistent with the
            validations for `MAX_EXPRESSION_LENGTH` and `MAX_OPERATOR_COUNT`.
            Consider updating either one to match the other and improve
            consistency.
        """
        invalid_exponent = max_exp_mag + 1
        with pytest.raises(
            CalculatorError,
            match="Exponent"
        ):
            calculate(f"2 ** {invalid_exponent}")

    def test_large_exponent_plus_base_fail(
            self,
            max_exp_for_base_check: int = MAX_EXPONENT_FOR_LARGE_BASE_CHECK,
            max_base_for_exp: int = MAX_BASE_FOR_LARGE_EXPONENT
    ) -> None:
        """
        Test upper limits for number exponentiation are enforced.

        Parameters
        ----------
            max_exp_for_base_check: int
                The magnitude of the exponent above which the base is
                checked for its magnitude using `max_base_for_exp`
            max_base_for_exp: int
                In case the exponent is considered "large" per the
                previous constant, the max value allowed for said
                exponent's base
        """
        invalid_large_exp = max_exp_for_base_check + 1
        invalid_large_base = max_base_for_exp + 1
        with pytest.raises(
            CalculatorError,
            match="is too large to raise to an exponent"
        ):
            calculate(f"{invalid_large_base} ** {invalid_large_exp}")

    @pytest.mark.parametrize("expression", [
        "2 ** 0.5",
        "3 ** 0.5",
        "5 ** 0.5"
    ])
    def test_max_decimal_places(
            self,
            expression: str,
            max_decimal_places: int = MAX_DECIMAL_PLACES
    ) -> None:
        """
        Test that results conform to the limit of `MAX_DECIMAL_PLACES`.

        Parameters
        ----------
            max_decimal_places: int
                # Digits to round off a response to

        Notes
        -----
            Use expressions that return irrational numbers as test cases.
        """
        result = str(calculate(expression))
        if "." in result:
            decimals = result.split(".")[1]
            assert len(decimals) <= max_decimal_places
