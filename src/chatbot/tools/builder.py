import inspect
import types


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
        doc = fn.__doc__ or ""
        registry[name] = fn

        # Map docstring param descriptions by name
        param_docs = {
            p.arg_name: p.description
            for p in doc.params
        } if "params" in doc and isinstance(doc, dict) else {}

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

        # Solve for function's description
        if isinstance(doc, str):
            description = doc
        elif isinstance(doc, dict):
            description = doc.get("short_description", "")
        else:
            doc = ""


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
