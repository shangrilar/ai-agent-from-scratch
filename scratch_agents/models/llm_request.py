from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ..types.contents import ContentItem


class LlmRequest(BaseModel):
    """Request object for LLM calls"""
    instructions: List[str] = Field(default_factory=list)
    contents: List[ContentItem] = Field(default_factory=list)
    tools_dict: Dict[str, Any] = Field(default_factory=dict)
    tool_choice: Optional[str] = None

    def add_instructions(self, instructions: List[str] | str):
        """Add instructions to the request"""
        if isinstance(instructions, str):
            self.instructions.append(instructions)
        else:
            self.instructions.extend(instructions)
        