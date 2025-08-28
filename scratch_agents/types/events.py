from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
from ..models.llm_response import LlmResponse
from .contents import ContentItem, ToolCall, ToolResult


class Event(LlmResponse):
    """Event wraps LlmResponse with additional metadata for tracking and identification"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str
    # When set, indicates a structured final tool is required for completion
    required_output_tool: Optional[str] = None
    
    def is_final_response(self) -> bool:
        """Check if this event represents a final response"""
        tool_calls = self.get_tool_calls()
        tool_results = self.get_tool_results()
        
        # If a structured final answer is required, consider it final when we have a successful
        # result from the required tool
        if self.required_output_tool:
            for tr in tool_results:
                if tr.name == self.required_output_tool and tr.status == "success":
                    return True
            return False
        
        return not tool_calls and not tool_results
    
    def get_tool_calls(self) -> List[ToolCall]:
        """Get all tool calls from the event"""
        return [item for item in self.content if isinstance(item, ToolCall)]
    
    def get_tool_results(self) -> List[ToolResult]:
        """Get all tool results from the event"""
        return [item for item in self.content if isinstance(item, ToolResult)]