"""AgentTool: wrap an Agent as a tool for use by other agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Type

from pydantic import BaseModel

from scratch_agent.tools.base import BaseTool
from scratch_agent.tools.helpers import format_tool_definition
from scratch_agent.context import ExecutionContext

if TYPE_CHECKING:
    from scratch_agent.agent import Agent


class AgentTool(BaseTool):
    """Adapter that wraps an Agent as a tool."""

    def __init__(
        self,
        agent: "Agent",
        input_schema: Type[BaseModel] | None = None,
    ):
        self.agent = agent
        self.input_schema = input_schema

        if input_schema:
            parameters = input_schema.model_json_schema()
            parameters.pop("$defs", None)
            parameters.pop("title", None)
        else:
            parameters = {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The task or question to delegate"
                    }
                },
                "required": ["request"]
            }

        tool_def = format_tool_definition(agent.name, agent.description, parameters)
        super().__init__(name=agent.name, description=agent.description, tool_definition=tool_def)

    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        if self.input_schema:
            validated = self.input_schema.model_validate(kwargs)
            request = validated.model_dump_json(exclude_none=True)
        else:
            request = kwargs.get("request", str(kwargs))
        result = await self.agent.run(request)
        return result.output
