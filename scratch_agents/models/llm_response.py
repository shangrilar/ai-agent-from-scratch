from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ..types.contents import ContentItem


class LlmResponse(BaseModel):
    """Response object from LLM calls"""
    content: List[ContentItem] = Field(default_factory=list)
    error_message: Optional[str] = None
    usage_metadata: Dict[str, Any] = Field(default_factory=dict)