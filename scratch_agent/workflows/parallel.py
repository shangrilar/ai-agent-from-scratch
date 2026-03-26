"""Parallel workflow: run agents concurrently."""

from __future__ import annotations

import asyncio
from typing import List

from scratch_agent.agent import Agent
from scratch_agent.context import AgentResult, ExecutionContext


class ParallelWorkflow(Agent):
    """Run agents in parallel and combine results."""

    def __init__(
        self,
        agents: List[Agent],
        name: str = "parallel_workflow",
    ):
        self.agents = agents
        self.name = name

    async def run(
        self,
        user_input: str | None = None,
        context: ExecutionContext | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> AgentResult:
        """Execute all agents concurrently."""
        if context is None:
            context = ExecutionContext()

        existing_event_count = len(context.events) if context else 0

        results = await asyncio.gather(
            *[agent.run(user_input, context=context, verbose=verbose) for agent in self.agents]
        )

        merged_context = ExecutionContext()
        if context is not None:
            for event in context.events:
                merged_context.add_event(event)

        seen_user_event = (context is not None)
        for result in results:
            new_events = result.context.events[existing_event_count:]
            for event in new_events:
                if event.author == "user":
                    if not seen_user_event:
                        merged_context.add_event(event)
                        seen_user_event = True
                else:
                    merged_context.add_event(event)

        # Combine outputs
        combined_output = "\n\n".join(
            f"[{agent.name}]\n{result.output}"
            for agent, result in zip(self.agents, results)
        )
        return AgentResult(output=combined_output, context=merged_context)
