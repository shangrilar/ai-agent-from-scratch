"""CH04 snapshot: Basic tool abstraction
Differences from final version:
  - requires_confirmation not present
  - sandbox_executable not present, get_source_code() not present
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from pydantic import BaseModel

from scratch_agent.tools.helpers import format_tool_definition, function_to_input_schema
from scratch_agent.context import ExecutionContext


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(
        self,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None,
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition

    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass

    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)


class FunctionTool(BaseTool):
    """Wraps a Python function as a BaseTool."""

    def __init__(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None,
    ):
        self.func = func
        self.needs_context = "context" in inspect.signature(func).parameters

        resolved_name = name or func.__name__
        resolved_desc = description or (func.__doc__ or "").strip()

        super().__init__(
            name=resolved_name,
            description=resolved_desc,
            tool_definition=tool_definition,
        )

        if self._tool_definition is None:
            self._tool_definition = self._generate_definition()

    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """Execute the wrapped function."""
        if self.needs_context:
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)

        if inspect.iscoroutine(result):
            return await result
        return result

    def _generate_definition(self) -> Dict[str, Any]:
        """Generate tool definition from function signature."""
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)


def tool(func=None, *, name=None, description=None):
    """Decorator to create a FunctionTool from a function."""
    def decorator(f):
        return FunctionTool(
            func=f,
            name=name,
            description=description,
        )

    if func is not None:
        return decorator(func)
    return decorator
