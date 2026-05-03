"""Memory tool with automatic injection of relevant past experiences (Listing 6.37)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scratch_agents.context import ExecutionContext
from scratch_agents.tools.base import BaseTool
from scratch_agents.types import Message

if TYPE_CHECKING:
    from scratch_agents.llm import LlmRequest
    from scratch_agents.memory.long_term import TaskMemory


class MemoryTool(BaseTool):
    """Tool that injects relevant past memories into the LLM request automatically."""

    def __init__(self):
        super().__init__(
            name="recall_memory",
            description=(
                "Search for past problem-solving records. "
                "Use this to check if similar problems were solved before."
            ),
            tool_definition=None,  # Automatic injection only
        )

    async def execute(self, context: ExecutionContext, query: str = "") -> str:
        """Search memories and return formatted results."""
        if context.memory_manager is None:
            return ""
        memories = await context.memory_manager.search(query, top_k=3)
        if not memories:
            return ""
        return self._format_memories(memories)

    def _format_memories(self, memories: list["TaskMemory"]) -> str:
        """Format memories for display."""
        results = []
        for i, mem in enumerate(memories, 1):
            status = "Correct" if mem.is_correct else "Incorrect"
            text = (
                f"[Record {i}]\n"
                f"- Problem: {mem.task_summary}\n"
                f"- Approach: {mem.approach}\n"
                f"- Answer: {mem.final_answer}\n"
                f"- Result: {status}"
            )
            if not mem.is_correct and mem.error_analysis:
                text += f"\n- Error analysis: {mem.error_analysis}"
            results.append(text)
        return "\n\n".join(results)

    async def process_llm_request(
        self,
        context: ExecutionContext,
        request: "LlmRequest",
    ) -> None:
        """Inject relevant memories before LLM call."""
        if context.memory_manager is None:
            return

        user_msgs = [
            c for c in request.contents
            if isinstance(c, Message) and c.role == "user"
        ]
        if not user_msgs:
            return

        result = await self.execute(context, user_msgs[-1].content)
        if not result:
            return

        request.append_instructions(
            "The following are records from similar problems solved in the past:\n"
            "<PAST_EXPERIENCES>\n"
            f"{result}\n"
            "</PAST_EXPERIENCES>\n"
            "Reference successful approaches and avoid approaches that led to failures."
        )
