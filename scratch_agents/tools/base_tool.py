from typing import Any, Dict, Type, Union
from abc import ABC, abstractmethod
import json
from .schema_utils import format_tool_definition


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
    
    @abstractmethod
    async def __call__(self, **kwargs) -> Any:
        return await self.execute(**kwargs)
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the execute method"
        )