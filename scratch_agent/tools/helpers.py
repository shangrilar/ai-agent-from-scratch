import inspect
import json
from typing import get_type_hints


def function_to_input_schema(func) -> dict:
    """Convert a function's signature to a JSON Schema for tool parameters.

    Inspects type hints and docstring to generate the schema.
    """
    try:
        hints = get_type_hints(func)
    except Exception:
        # Fallback for closures where get_type_hints can't resolve annotations
        hints = {
            name: param.annotation
            for name, param in inspect.signature(func).parameters.items()
            if param.annotation is not inspect.Parameter.empty
        }
    sig = inspect.signature(func)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "context"):
            continue

        prop = {}
        hint = hints.get(name)

        if hint == str:
            prop["type"] = "string"
        elif hint == int:
            prop["type"] = "integer"
        elif hint == float:
            prop["type"] = "number"
        elif hint == bool:
            prop["type"] = "boolean"
        elif hint == list or (hasattr(hint, "__origin__") and hint.__origin__ is list):
            prop["type"] = "array"
            # Try to get item type
            if hasattr(hint, "__args__") and hint.__args__:
                item_type = hint.__args__[0]
                if item_type == str:
                    prop["items"] = {"type": "string"}
                elif item_type == int:
                    prop["items"] = {"type": "integer"}
                elif hasattr(item_type, "model_json_schema"):
                    prop["items"] = item_type.model_json_schema()
        elif hasattr(hint, "model_json_schema"):
            # Pydantic model
            prop = hint.model_json_schema()
        else:
            prop["type"] = "string"

        # Add description from docstring if available
        prop["description"] = f"Parameter: {name}"

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def format_tool_definition(name: str, description: str, parameters: dict) -> dict:
    """Format a tool definition in the OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    }


def function_to_tool_definition(func) -> dict:
    """Convert a Python function to an OpenAI-format tool definition.

    Uses the function name, docstring, and type hints.
    """
    name = func.__name__
    description = inspect.getdoc(func) or f"Function: {name}"
    parameters = function_to_input_schema(func)
    return format_tool_definition(name, description, parameters)


def tool_execution(tool_box: dict, tool_call) -> str:
    """Execute a tool call using a tool_box mapping.

    Args:
        tool_box: Dict mapping tool names to callables
        tool_call: Tool call object with function.name and function.arguments
    """
    func_name = tool_call.function.name
    if func_name not in tool_box:
        return f"Error: Unknown tool '{func_name}'"

    try:
        args = json.loads(tool_call.function.arguments)
        result = tool_box[func_name](**args)
        return str(result)
    except Exception as e:
        return f"Error executing {func_name}: {str(e)}"
