from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from ..types.events import Event
from ..types.contents import Message
from pydantic import BaseModel
import uuid

@dataclass
class ExecutionContext:
    """Manages the execution state of an agent throughout its lifecycle."""
    
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    events: List[Event] = field(default_factory=list)
    user_input: Optional[Message] = None
    current_step: int = 0
    
    state: Dict[str, Any] = field(default_factory=dict)
    
    final_result: str | BaseModel = None
    
    def add_event(self, event: Event):
        """Add an event to the history"""
        self.events.append(event)
    
    def increment_step(self):
        self.current_step += 1