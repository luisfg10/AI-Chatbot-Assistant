import inspect
import types
from collections.abc import Callable
from typing import Any

from docstring_parser import parse as parse_docstring
from docstring_parser.common import Docstring


class DocstringError(ValueError):
    """
    Custom exception for raising docstring-related errors.

    Raised when a tool function's docstring is missing, incomplete, or
    doesn't match its actual signature.
    """


def _validate_docstring(
        fn: Callable[[Any], Any],
        sig: inspect.Signature,
        parsed_doc: dict | Docstring
) -> None:
    """
    Validate a function's docstring is complete and well-formatted.

    Checks that a function's docstring is consistent with its actual
    signature, or raises DocstringError with a helpful message otherwise.
    This is necessary because `docstring_parser` fails silently on malformed
    numpydoc/Google/reST syntax.

    Parameters
    ----------
    fn: Callable
        The function to inspect.
    sig: inspect.Signature
        The function's signature, obtained using the inspect library.
    parsed_doc: dict | Docstring
        Dict containing the function's parsed docstring as keys,
        or Docstring object with the docstring parameters as attributes.
    """
    name = fn.__name__

    # Invariant 1: the function must have a 1-line summary
    if not parsed_doc.short_description:
        raise DocstringError(
            f"Function '{name}' has no docstring summary. Add a one-line "
            f"description of what the tool does."
        )

    # Invariant 2: every parameter in the function's real signature must have a
    # matching, non-empty entry in the docstring's Parameters section
    sig_param_names = set(sig.parameters.keys())
    doc_param_names = {p.arg_name for p in parsed_doc.params}

    missing_from_doc = sig_param_names - doc_param_names
    if missing_from_doc:
        raise DocstringError(
            f"Function '{name}' has parameter(s) {sorted(missing_from_doc)} "
            "that are not documented properly in the docstring. "
            "Make sure to document using consistent numpy-style formatting."
        )

    # Invariant 3: A documented param that doesn't exist in signature
    extra_in_doc = doc_param_names - sig_param_names
    if extra_in_doc:
        raise DocstringError(
            f"Function '{name}' documents parameter(s) {sorted(extra_in_doc)} "
            "that don't exist in its actual signature. Check for typos or "
            "stale documentation."
        )

    # Invariant 4: every documented param must have a description
    empty_descriptions = [
        p.arg_name for p in parsed_doc.params
        if not (p.description or "").strip()
    ]
    if empty_descriptions:
        raise DocstringError(
            f"Function '{name}' has parameter(s) {sorted(empty_descriptions)} "
            f"documented with no description text."
        )


def build_tools(functions: list) -> tuple[dict, list]:
    """
    Build a tool registry and schema list from a list of functions.

    Reuses each function's docstring as description when building the
    agent-facing schema, so it's essential for it to be clear and concise.

    Parameters
    ----------
    functions: list
        A list of functions to be used as tools.

    Returns
    -------
    registry: dict
        A dict mapping a function's name to its object.
    schemas: list[dict]
        A list containing the schemas for all of the tools
        to be used.

    Examples
    --------
    >>> from src.chatbot.tools.definitions import get_current_date
    >>> functions_list = [get_current_date]
    >>> registry, schema = build_tools(functions_list)
    >>> print(registry)
        {'get_current_date': <function get_current_date at 0x000001HJRY>}
    >>> print(schema)
        [{
            'type': 'function',
            'function': {
                'name': 'get_current_date',
                'description': "Get today's date in YYYY-MM-DD format.",
                'parameters': {
                    'type': 'object',
                    'properties': {},
                    'required': []
                }
            }
        }]
    """
    # Define return objects
    registry = {}
    schemas = []

    # Map Python type annotations to JSON Schema types
    type_map = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
    }

    for fn in functions:
        # Check object is actually a function
        if not isinstance(fn, types.FunctionType):
            raise ValueError(f"Object '{fn}' must be a function.")

        # Get metadata from the function
        name = fn.__name__
        sig = inspect.signature(fn)
        registry[name] = fn

        # Parse the docstring (numpy, Google, or reST style all work) into
        # a structured object with separate fields for the summary, the
        # extended description, and each parameter's description
        parsed_doc = parse_docstring(inspect.getdoc(fn) or "")

        # Enforce docstring/signature consistency before using any of
        # this function's documentation to build its schema. Raises
        # DocstringError immediately if something doesn't line up.
        _validate_docstring(fn, sig, parsed_doc)

        # Map docstring param descriptions by name
        param_docs = {
            p.arg_name: p.description or ""
            for p in parsed_doc.params
        }

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            annotation = param.annotation
            json_type = type_map.get(annotation, "string")
            properties[param_name] = {
                "type": json_type,
                "description": param_docs.get(param_name, "")
            }
            # If no default value, it's required
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        # Build the function-level description from the docstring's
        # short + long description only — this deliberately excludes the
        # Parameters/Returns sections, since those are represented
        # separately in the schema's "properties" above
        description = parsed_doc.short_description or ""
        if parsed_doc.long_description:
            description += "\n" + parsed_doc.long_description

        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })

    return registry, schemas
