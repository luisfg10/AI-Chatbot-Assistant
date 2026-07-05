import inspect
import types

from docstring_parser import parse as parse_docstring


class DocstringError(ValueError):
    """
    Raised when a tool function's docstring is missing, incomplete, or
    doesn't match its actual signature.
    
    Catching this separately from a plain ValueError makes it easy
    to distinguish bad documentation and fix them.
    """


def _validate_docstring(
        fn,
        sig: inspect.Signature,
        parsed_doc
) -> None:
    """
    Check that a function's docstring is complete and consistent with its
    actual signature, and raise DocstringError with a specific, actionable
    message otherwise.

    This is necessary because `docstring_parser` fails *silently* on malformed
    numpydoc/Google/reST syntax.
    """
    name = fn.__name__

    # Check 1: the function must have a short summary line.
    if not parsed_doc.short_description:
        raise DocstringError(
            f"Function '{name}' has no docstring summary. Add a one-line "
            f"description of what the tool does."
        )

    # Check 2: every parameter in the function's real signature must have a
    # matching, non-empty entry in the docstring's Parameters section.
    sig_param_names = set(sig.parameters.keys())
    doc_param_names = {p.arg_name for p in parsed_doc.params}

    missing_from_doc = sig_param_names - doc_param_names
    if missing_from_doc:
        raise DocstringError(
            f"Function '{name}' has parameter(s) {sorted(missing_from_doc)} "
            "that are not documented properly in the docstring. "
            "Make sure to document using consistent numpy-style formatting."
        )

    # Check 3: A documented parameter that doesn't
    # actually exist in the function signature
    extra_in_doc = doc_param_names - sig_param_names
    if extra_in_doc:
        raise DocstringError(
            f"Function '{name}' documents parameter(s) {sorted(extra_in_doc)} "
            "that don't exist in its actual signature. Check for typos or "
            "stale documentation."
        )

    # Check 4: every documented parameter must have actual description
    # text, not just a name with an empty/missing body.
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

    Uses each function's docstring as description when building the
    schema, so it's essential it be concise and informative.

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
        # Sanity check: object must be a function
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
