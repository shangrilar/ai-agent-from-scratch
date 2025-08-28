from typing import Literal, Optional, Union, List
from pydantic import BaseModel, Field
from datetime import datetime

class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict
    
class ToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list

class Message(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["developer", "user", "assistant"]
    content: str

ContentItem = Union[Message, ToolCall, ToolResult]