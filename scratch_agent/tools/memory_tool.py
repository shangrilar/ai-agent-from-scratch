"""Memory tool for long-term memory search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scratch_agent.tools.base import BaseTool
from scratch_agent.context import ExecutionContext

if TYPE_CHECKING:
    from scratch_agent.memory.long_term import TaskMemoryManager


class MemoryTool(BaseTool):
    """Tool for searching long-term task memories."""

    def __init__(self, memory_manager: "TaskMemoryManager"):
        self.memory_manager = memory_manager
        super().__init__(
            name="search_memory",
            description="Search long-term memory for relevant past task experiences.",
            tool_definition={
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search long-term memory for relevant past task experiences. Use this when you encounter a problem similar to ones you may have solved before.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query describing the task or problem",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
        )

    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """Search memories matching the query."""
        query = kwargs.get("query", "")
        memories = await self.memory_manager.search(query)

        if not memories:
            return "No relevant memories found."

        results = []
        for i, mem in enumerate(memories, 1):
            results.append(
                f"Memory {i}:\n"
                f"  Task: {mem.task_summary}\n"
                f"  Approach: {mem.approach}\n"
                f"  Answer: {mem.final_answer}"
            )
        return "\n\n".join(results)
