from typing import Callable, Union, Dict, Any
from ch3_06_tool_abstraction import FunctionTool

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
    
    # Handle both @tool and @tool() usage
    if func is not None:
        return decorator(func)
    return decorator

if __name__ == "__main__":
    def search_web(query: str) -> str:
        """Search for information on the web"""
        return f"{query}_result"

    search_tool_v1 = FunctionTool(search_web)  

    @tool  
    def search_web(query: str) -> str:
        """Search for information on the web"""
        return f"{query}_result"

    @tool(name="internet_search",  
        description="Query the internet for latest information")  
    def search_web_custom(query: str) -> str:
        """Search for information on the web"""
        return f"{query}_result"

    print(search_tool_v1.tool_definition)
    print(search_web.tool_definition)
    print(search_web_custom.tool_definition)