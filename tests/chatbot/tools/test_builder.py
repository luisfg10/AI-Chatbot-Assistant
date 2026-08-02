import inspect
import json

import pytest

from src.chatbot.tools import definitions
from src.chatbot.tools.builder import DocstringError, build_tools

# ------------------------------------------------------------------
# Testing Utils


def _get_definitions_tools() -> list:
    """
    Discover every function defined inside definitions.py.

    Assumes that all defined functions inside the file are agent-facing
    tools and as such should be able to go through the tool-building
    function successfully.

    Useful to avoid having to manually update the tools in several places
    each time they're modified.
    """
    return [
        # Check function objects inside definitions
        fn for _, fn in inspect.getmembers(
            definitions,
            inspect.isfunction
        )
        # Check the function was originally defined inside definitions
        if fn.__module__ == definitions.__name__
    ]


def _valid_fn(name: str, age: int) -> str:
    """
    Greet a person by name and age.

    This is used as an example of a func with a valid docstring.

    Parameters
    ----------
    name: str
        The person's name.
    age: int
        The person's age.
    """
    return f"Hello {name}, age {age}"


# Function with missing summary in docstring
def _no_summary_fn(x: int) -> int:
    """Parameters
    ----------
    x: int
        A number.
    """
    return x


def _missing_param_doc_fn(x: int, y: int) -> int:
    """
    Add two numbers.

    Example of a func with a param missing explanation in docstring (y).

    Parameters
    ----------
    x: int
        The first number.
    """
    return x + y


def _extra_param_doc_fn(x: int) -> int:
    """
    Double a number.

    Example of a func with a documented param that doesn't exist.

    Parameters
    ----------
    x: int
        The number to double.
    y: int
        Doesn't exist.
    """
    return x * 2


def _empty_param_desc_fn(x: int) -> int:
    """
    Triple a number.

    Example of a func that doesn't describe params properly.

    Parameters
    ----------
    x: int
    """
    return x * 3


def _mixed_signature_fn(
        a: int,
        b: float = 1.0,
        c: list | None = None
) -> str:
    """
    Combine values with an optional and an unmapped-type param.

    Used for testing that reequired and optional parameters are
    correctly mapped when building the tool schema.

    Parameters
    ----------
    a: int
        Required int.
    b: float
        Optional float.
    c: list | None
        Not in the builder's type map, defaults to None.
    """
    return f"{a}-{b}-{c}"

# ------------------------------------------------------------------
# Testing Class


class TestBuilder:
    """Test the utils inside `builder.py`."""

    @pytest.mark.parametrize("fn, match", [
        (_no_summary_fn, "has no docstring summary"),
        (_missing_param_doc_fn, "not documented properly"),
        (_extra_param_doc_fn, "don't exist in its actual signature"),
        (_empty_param_desc_fn, "documented with no description text"),
    ])
    def test_invalid_docstrings_fail(self, fn: callable, match: str) -> None:
        """
        Test functions with malformed docstrings raise DocstringError.

        Fails if either DocstringError is not raised, or it doesn't
        contain the expected message.

        Parameters
        ----------
            fn: callable
                The function to test.
            match: str
                String to match inside the exception message.
        """
        with pytest.raises(DocstringError, match=match):
            build_tools([fn])

    def test_valid_docstring_passes(self) -> None:
        """Test a properly documented, consistent function builds cleanly."""
        registry, schemas = build_tools([_valid_fn])
        assert registry["_valid_fn"] is _valid_fn
        assert len(schemas) == 1

    @pytest.mark.parametrize("bad_obj", ["not a function", 123, object()])
    def test_non_function_object_fails(self, bad_obj: object) -> None:
        """Test passing non-function objects raises ValueError."""
        with pytest.raises(ValueError, match="must be a function"):
            build_tools([bad_obj])

    def test_all_definitions_tools_build(self) -> None:
        """Test every tool defined in definitions.py builds correctly."""
        functions = _get_definitions_tools()
        registry, schemas = build_tools(functions)
        assert set(registry) == {fn.__name__ for fn in functions}
        assert len(schemas) == len(functions)

    def test_schema_and_registry_are_well_formed(self) -> None:
        """Test mixed-signature functions build correctly."""
        registry, schemas = build_tools([_mixed_signature_fn])

        # Check registry's contents
        assert registry["_mixed_signature_fn"] is _mixed_signature_fn

        # Check schema's contents
        schema = schemas[0]
        assert schema["type"] == "function"
        assert set(schema["function"]) == {"name", "description", "parameters"}
        params = schema["function"]["parameters"]
        assert params["required"] == ["a"]  # only non-default params
        assert params["properties"]["a"]["type"] == "integer"
        assert params["properties"]["b"]["type"] == "number"
        assert params["properties"]["c"]["type"] == "string"  # unmapped-type fallback
        json.dumps(schema)  # Check JSON convertability
