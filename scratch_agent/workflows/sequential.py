"""Sequential workflow: run agents one after another."""

from __future__ import annotations

from typing import List

from scratch_agent.agent import Agent
from scratch_agent.context import AgentResult, ExecutionContext


class SequentialWorkflow(Agent):
    """Run agents sequentially, passing context from one to the next."""

    def __init__(
        self,
        agents: List[Agent],
        name: str = "sequential_workflow",
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
        """Execute all agents in sequence."""
        if context is None:
            context = ExecutionContext()

        result = None
        for i, agent in enumerate(self.agents):
            if context is not None:
                context.final_result = None
                context.current_step = 0

            if i == 0:
                result = await agent.run(
                    user_input=user_input,
                    context=context,
                    verbose=verbose,
                )
            else:
                result = await agent.run(
                    context=context,
                    verbose=verbose,
                )
            context = result.context

        return result
