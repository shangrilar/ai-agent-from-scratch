"""Agent transfer tool for multi-agent routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from scratch_agent.tools.base import FunctionTool, tool
from scratch_agent.context import ExecutionContext

if TYPE_CHECKING:
    from scratch_agent.agent import Agent


def create_transfer_tool(target_agents: List["Agent"]) -> FunctionTool:
    """Create transfer tool from list of transferable agents."""

    # Compose agent info
    target_names = [agent.name for agent in target_agents]
    agent_descriptions = []
    for agent in target_agents:
        desc = agent.description or agent.instructions[:100].replace('\n', ' ')
        if len(desc) > 100:
            desc = desc[:100] + "..."
        agent_descriptions.append(f"  - {agent.name}: {desc}")

    agent_info = "\n".join(agent_descriptions)

    @tool(
        name="transfer_to_agent",
        description=f"""Transfers work to another agent.

Use this tool when the current question belongs to another agent's specialty.

Available agents:
{agent_info}
"""
    )
    def transfer_to_agent(context: ExecutionContext, agent_name: str) -> str:
        """Agent transfer tool."""
        if agent_name not in target_names:
            return f"Error: '{agent_name}' is not valid. Available: {target_names}"

        # Only apply first transfer request
        if context.transfer_to is None:
            context.transfer_to = agent_name
            return f"Transferring to {agent_name}..."
        else:
            return f"Transfer already requested to {context.transfer_to}"

    # Add enum constraint
    transfer_to_agent.tool_definition["function"]["parameters"]["properties"]["agent_name"]["enum"] = target_names

    return transfer_to_agent
