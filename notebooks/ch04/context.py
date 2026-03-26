"""CH04 snapshot: Basic context
Differences from final version:
  - PendingToolCall/ToolConfirmation not present
  - ExecutionContext session/code_env/transfer_to not present
  - AgentResult status/pending_tool_calls not present
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from scratch_agent.types import Event


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None

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
