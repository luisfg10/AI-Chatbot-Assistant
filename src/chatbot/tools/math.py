r"""
Safe arithmetic calculator tool for AI agents.

Receives a string containing a math operation as input, and
performs several safety checks to ensure that:
    * The string is in fact a math expression with numbers and
    operators
    * The expression is not overly long and will not result in an
    overly-expensive computation that might freeze or crash the program

=================================================
Operation Safety

    Instead of running string as code, this tool:

    1. Uses Python's built-in `ast` module to parse the string into a tree
    structure that describes its grammar without executing anything.
    For example, the expression "3 * (8 + 1)" gets broken down into a tree:

                BinOp(*)
                /      \\
                3        BinOp(+)
                        /      \\
                        8        1

    2. Walks the tree node by node, and only allows specific pieces to exist:
    numbers, and the +, -, *, /, ** operators. If any other node type is found
    (function call, variable name, attribute access), the expression is not
    evaluated and an error is raised, guaranteeing that only basic arithmetic
    can be performed.

=================================================
Configuration Variables

    Additional guardrails are also in place to reject overly large arithemtic
    operations that may cause the application to freeze or crash, which are:

    * MAX_EXPRESSION_LENGTH: # of characters allowed in the expression.
    * MAX_OPERATOR_COUNT
    * MAX_EXPONENT_MAGNITUDE: Max value for exponents, independent of its base
    * MAX_DECIMAL_PLACES: # Digits to round off a response to
    * MAX_EXPONENT_FOR_LARGE_BASE_CHECK: The magnitude of the exponent above which
    the base is checked for its magnitude using MAX_BASE_FOR_LARGE_EXPONENT

"""

import ast
import operator

Number = int | float

# ------------------------------------------------------------------
# Config

# Standalone constants
MAX_EXPRESSION_LENGTH = 200
MAX_OPERATOR_COUNT = 25
MAX_EXPONENT_MAGNITUDE = 1000
MAX_DECIMAL_PLACES = 6

# Mutually-dependant
MAX_EXPONENT_FOR_LARGE_BASE_CHECK = 100
MAX_BASE_FOR_LARGE_EXPONENT = 10_000


class CalculatorError(Exception):
    """
    Catch-all exception used for any problem with the expression.

    Used for invalid syntax, disallowed operations, or a result that
    breaks one of the safety limits above.
    """


# ------------------------------------------------------------------
# Allowed Math Operators Whitelist

_BINOPS = {
    ast.Add: operator.add,       # +
    ast.Sub: operator.sub,       # -
    ast.Mult: operator.mul,      # *
    ast.Div: operator.truediv,   # /
    ast.Pow: operator.pow,       # **
}

# Unary operators: operators that work on a single value, e.g. "-5" or "+5"
# (the minus/plus sign directly in front of a number, not between two numbers)
_UNARYOPS = {
    ast.UAdd: operator.pos,   # +5 (rare, but valid: "positive 5")
    ast.USub: operator.neg,   # -5
}

# ------------------------------------------------------------------
# Auxiliary Functions (not agent-facing)


def _count_operators(node: ast.AST) -> int:
    """
    Walk a parsed expression tree and count its operator nodes.

    Counts BinOp and UnaryOp nodes. Used to enforce
    MAX_OPERATOR_COUNT before we bother evaluating anything.
    """
    count = 0
    for child in ast.walk(node):
        if isinstance(child, (ast.BinOp, ast.UnaryOp)):
            count += 1
    return count


def _eval_node(node: ast.AST) -> Number:
    """
    Recursively evaluate one node of the parsed expression tree.

    If any node type that isn't explicitly handled below is encountered,
    a CalculatorError is raised instead of executing it.
    """
    # Case: the top-level wrapper Python adds when parsing in "eval" mode.
    # `ast.parse(expr, mode="eval")` always returns an `ast.Expression`
    # object as the root, with the actual content stored in `.body`.
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    # Case: a literal number, like `3`, `8`, or `2.5`.
    # In modern Python, all literal constants (numbers, strings,
    # True/False, None, etc.) are represented as `ast.Constant`.
    # Only numeric constants are allowed.
    if isinstance(node, ast.Constant):
        if (
            isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return node.value
        raise CalculatorError(
            f"Only numeric literals allowed, got: {node.value!r}"
        )

    # Case: a binary operation, like `3 + 4`, `8 * 2`, `10 / 5`, `2 ** 3`.
    # An ast.BinOp node has three parts: `.left` (left side), `.op` (which
    # operator), and `.right` (right side). Evaluate left and right
    # recursively, then apply the operatos
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        # Catch non-allowed operators
        if op_type not in _BINOPS:
            raise CalculatorError(f"Operator not allowed: {op_type.__name__}")

        left = _eval_node(node.left)
        right = _eval_node(node.right)

        # Safety checks for exponentiation
        if op_type is ast.Pow:
            if abs(right) > MAX_EXPONENT_MAGNITUDE:
                raise CalculatorError(
                    f"Exponent {right} exceeds maximum allowed magnitude "
                    f"({MAX_EXPONENT_MAGNITUDE})"
                )
            if (
                abs(right) > MAX_EXPONENT_FOR_LARGE_BASE_CHECK
                and abs(left) > MAX_BASE_FOR_LARGE_EXPONENT
            ):
                raise CalculatorError(
                    f"Base {left} is too large to raise to an exponent "
                    f"greater than {MAX_EXPONENT_FOR_LARGE_BASE_CHECK}"
                )

        try:
            return _BINOPS[op_type](left, right)
        except ZeroDivisionError:
            raise CalculatorError("Division by zero") from None
        except OverflowError:
            raise CalculatorError("Result too large to compute") from None

    # Case: a unary operation, like `-5`
    # An ast.UnaryOp object has `.op` (which sign) and `.operand`
    # (the value it applies to).
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARYOPS:
            raise CalculatorError(
                f"Unary operator not allowed: {op_type.__name__}"
            )
        return _UNARYOPS[op_type](_eval_node(node.operand))

    # --- Case: anything else.
    # e.g., function call (`ast.Call`), a variable name (`ast.Name`),
    # attribute access (`ast.Attribute`, e.g. `os.system`), etc.
    raise CalculatorError(
        f"Disallowed expression element: {type(node).__name__}"
    )


def _format_result(result: Number) -> Number:
    """Enforce formatting rules for numeric results with decimals."""
    # Tidy up decimals.
    if isinstance(result, float):
        result = round(result, MAX_DECIMAL_PLACES)
        # If rounding produced a whole number (e.g. 4.0), return it as a
        # plain int (4) instead — it reads more naturally in a response.
        if result.is_integer():
            result = int(result)

    return result


def calculate(expression: str) -> Number:
    """
    Safely evaluate a basic arithmetic expression and return a number.

    Supports: + - * / ** and parentheses, with int/float numbers.

    Raises CalculatorError if:
      - the input isn't a valid non-empty string,
      - the expression is too long,
      - the expression contains too many operators,
      - the expression has invalid syntax,
      - the expression contains anything other than numbers and the
        allowed arithmetic operators,
      - the exponent or base in a `**` operation is too large,
      - a division by zero is attempted,
      - the final result is too large (too many digits).
    """
    # Basic input validation
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("Expression must be a non-empty string")

    expression = expression.lower()

    # Attempt to "translate" characters to valid, evaluable operators
    char_mapping = str.maketrans({
        "x": "*",
        "^": "**"
    })
    expression = expression.translate(char_mapping)

    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(
            f"Expression is {len(expression)} characters long, which exceeds "
            f"the maximum allowed ({MAX_EXPRESSION_LENGTH})"
        )

    # Parse the string into a tree structure using ast
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(
            f"Invalid expression syntax: {e}"
        ) from e

    # --- Enforce the operator-count limit before evaluating anything.
    operator_count = _count_operators(tree)
    if operator_count > MAX_OPERATOR_COUNT:
        raise CalculatorError(
            f"Expression has {operator_count} operators, which exceeds the "
            f"maximum allowed ({MAX_OPERATOR_COUNT})"
        )

    # --- Walk the tree and compute the actual result.
    result = _eval_node(tree)

    # --- Apply final formatting/size checks before returning.
    return _format_result(result)
