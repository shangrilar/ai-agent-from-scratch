"""A2A server adapter for exposing an Agent as an A2A service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scratch_agent.agent import Agent


class AgentExecutor:
    """Base class for A2A agent executors."""

    async def execute(self, context: Any, event_queue: Any) -> None:
        raise NotImplementedError


class MathAgentExecutor(AgentExecutor):
    """Example A2A executor wrapping an Agent."""

    def __init__(self, agent: "Agent"):
        self.agent = agent

    async def execute(self, context: Any, event_queue: Any) -> None:
        """Execute the agent and push results to the event queue."""
        # Extract user message from A2A context
        user_input = ""
        if hasattr(context, "message"):
            for part in context.message.get("parts", []):
                if part.get("type") == "text":
                    user_input = part["text"]
                    break

        result = await self.agent.run(user_input=user_input)

        # Push result to event queue
        if event_queue and result.output:
            await event_queue.put({
                "type": "artifact",
                "parts": [{"type": "text", "text": str(result.output)}],
            })
