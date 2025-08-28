from typing import Callable, Union, Dict, Any
from .function_tool import FunctionTool

def tool(
    func: Callable = None,
    *,
    name: str = None,
    description: str = None,
    tool_definition: Union[Dict[str, Any], str] = None
) -> Union[Callable, FunctionTool]:
    
    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(
            func=f,
            name=name,
            description=description,
            tool_definition=tool_definition
        )
    
    if func is not None:
        return decorator(func)
    return decorator