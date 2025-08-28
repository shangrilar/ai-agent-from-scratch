from abc import abstractmethod
from pydantic import BaseModel
from .llm_request import LlmRequest

class BaseLlm(BaseModel):
    """Abstract base class for LLM implementations"""
    
    model: str
    
    @abstractmethod
    async def generate(self, request: LlmRequest):
        pass