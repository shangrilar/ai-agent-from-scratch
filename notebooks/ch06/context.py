"""CH06 snapshot: Context + session + Human-in-the-loop
Changes from CH04:
  - ExecutionContext session, session_manager fields added
  - AgentResult status, pending_tool_calls fields added
  - PendingToolCall, ToolConfirmation new types
  - code_env, transfer_to not yet added
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from scratch_agents.types import Event, ToolCall


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None
    # NEW: CH06 session
    session: Optional[Any] = None
    session_manager: Optional[Any] = None

    def add_event(self, event: Event):
        """Append an event to the execution history."""
        self.events.append(event)

    def increment_step(self):
        """Move to the next execution step."""
        self.current_step += 1


@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: Any  # str | BaseModel
    context: ExecutionContext
    # NEW: CH06
    status: str = "complete"  # "complete" | "pending" | "error"
    pending_tool_calls: list = field(default_factory=list)


# NEW: CH06 Human-in-the-loop types
class PendingToolCall(BaseModel):
    """A tool call awaiting user confirmation."""
    tool_call: ToolCall
    confirmation_message: str


class ToolConfirmation(BaseModel):
    """User's response to a pending tool call."""
    tool_call_id: str
    approved: bool
    modified_arguments: dict | None = None
