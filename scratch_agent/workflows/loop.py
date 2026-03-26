"""Loop workflow: run agents repeatedly until a stop condition is met."""

from __future__ import annotations

from typing import Callable, List

from scratch_agent.agent import Agent
from scratch_agent.context import AgentResult, ExecutionContext

StopCondition = Callable[[AgentResult, int], bool]


class LoopWorkflow(Agent):
    """Run agents in a loop until stop condition is met."""

    def __init__(
        self,
        agents: List[Agent],
        stop_condition: StopCondition | None = None,
        max_iterations: int = 10,
        name: str = "loop_workflow",
    ):
        self.agents = agents
        self.stop_condition = stop_condition
        self.max_iterations = max_iterations
        self.name = name

    async def run(
        self,
        user_input: str | None = None,
        context: ExecutionContext | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> AgentResult:
        """Execute agents in a loop."""
        if context is None:
            context = ExecutionContext()

        result = None
        is_first_agent = True

        for iteration in range(1, self.max_iterations + 1):
            for agent in self.agents:
                if context is not None:
                    context.final_result = None
                    context.current_step = 0

                if is_first_agent:
                    result = await agent.run(
                        user_input=user_input,
                        context=context,
                        verbose=verbose,
                    )
                    is_first_agent = False
                else:
                    result = await agent.run(
                        context=context,
                        verbose=verbose,
                    )
                context = result.context

            if result and self.stop_condition and self.stop_condition(result, iteration):
                break

        return result
