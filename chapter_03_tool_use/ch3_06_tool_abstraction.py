from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Type, Callable
import asyncio
import inspect
import json
from ch3_04_tool_definition import format_tool_definition, function_to_input_schema

# Listing 3.27
class BaseTool(ABC):
    
    def __init__(
        self, 
        name: str = None, 
        description: str = None, 
        tool_definition: Union[Dict[str, Any], str] = None,
        pydantic_input_model: Type = None
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self.pydantic_input_model = pydantic_input_model
        
        if isinstance(tool_definition, str):
            self._tool_definition = json.loads(tool_definition)
        elif tool_definition is not None:
            self._tool_definition = tool_definition
        else:
            self._tool_definition = None  # Generate later

    # Listing 3.28
    @property
    def tool_definition(self) -> Dict[str, Any]:
        if self._tool_definition is None:  #A
            self._tool_definition = self._generate_definition()  #A
        return self._tool_definition
    
    def _generate_definition(self) -> Dict[str, Any]:
        if self.pydantic_input_model:
            try:
                from pydantic import BaseModel
                if issubclass(self.pydantic_input_model, BaseModel):
                    parameters = self.pydantic_input_model.model_json_schema()
                    return format_tool_definition(
                        self.name, self.description, parameters
                    )
            except ImportError:
                pass
        # Subclasses should override this method or provide tool_definition
        raise NotImplementedError(
            f"{self.__class__.__name__} must either provide a tool_definition, "
            f"pydantic_input_model, or override _generate_definition()"
        )

# Listing 3.29
class FunctionTool(BaseTool):
    
    def __init__(
        self, 
        func: Callable, 
        name: str = None, 
        description: str = None,
        tool_definition: Union[Dict[str, Any], str] = None
    ):
        self.func = func
        self.pydantic_input_model = self._detect_pydantic_model(func)  #A
        
        name = name or func.__name__  #B
        description = description or (func.__doc__ or "").strip()  #B
        
        super().__init__(
            name=name, 
            description=description, 
            tool_definition=tool_definition,
            pydantic_input_model=self.pydantic_input_model
        )
    
    # Listing 3.30
    async def execute(self, **kwargs) -> Any:
        if self.pydantic_input_model:
            args = (self.pydantic_input_model.model_validate(kwargs),)
            call_kwargs = {}
        else:
            args = ()
            call_kwargs = kwargs
        
        if inspect.iscoroutinefunction(self.func):
            return await self.func(*args, **call_kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.func(*args, **call_kwargs)
            )
    
    # Listing 3.31
    def _generate_definition(self) -> Dict[str, Any]:
        if self.pydantic_input_model:
            return super()._generate_definition()
        
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)

    # Listing 3.32
    def _detect_pydantic_model(self, func: Callable) -> Optional[Type]:
        try:
            from pydantic import BaseModel
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            
            if len(params) == 1 and params[0].annotation != inspect._empty:
                param_type = params[0].annotation
                if isinstance(param_type, type) and issubclass(param_type, BaseModel):
                    return param_type
        except ImportError:
            pass
        return None

if __name__ == "__main__":
    def search_web(query: str) -> str:
        """Search for information on the web"""
        # Actual search logic
        return f"Search results: {query}"

    search_tool = FunctionTool(search_web)

    print(type(search_tool))   
    print(search_tool.description)
    print(search_tool.tool_definition)
