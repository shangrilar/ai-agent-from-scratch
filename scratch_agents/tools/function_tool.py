from typing import Any, Dict, Type, Union, Callable, Optional
import inspect
import asyncio
from .base_tool import BaseTool
from .schema_utils import format_tool_definition, function_to_input_schema

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
    
    def _generate_definition(self) -> Dict[str, Any]:
        if self.pydantic_input_model:
            return super()._generate_definition()
        
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)

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