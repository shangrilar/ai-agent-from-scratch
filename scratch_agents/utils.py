"""Utility helpers for displaying agent execution traces."""

from scratch_agents.types import Message, ToolCall, ToolResult


def display_trace(context) -> None:
    """Print a readable trace of events recorded in an ExecutionContext."""
    for index, event in enumerate(context.events, start=1):
        print(f"\n[{index}] {event.author}")
        for item in event.content:
            if isinstance(item, Message):
                print(f"  {item.role}: {item.content}")
            elif isinstance(item, ToolCall):
                print(f"  tool call: {item.name}({item.arguments})")
            elif isinstance(item, ToolResult):
                preview = str(item.content[0]) if item.content else ""
                print(f"  tool result: {item.name} -> {preview[:500]}")
