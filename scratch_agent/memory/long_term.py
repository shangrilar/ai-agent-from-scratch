"""Long-term memory with ChromaDB for task memory storage and retrieval."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from scratch_agent.context import ExecutionContext
from scratch_agent.types import ContentItem, Event, Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from scratch_agent.llm import LlmClient


class TaskMemory(BaseModel):
    """A memory of a completed task."""
    task_summary: str
    approach: str
    final_answer: str
    is_correct: bool | None = None
    error_analysis: str = ""

    def to_embedding_text(self) -> str:
        """Convert to text suitable for embedding."""
        parts = [
            f"Task: {self.task_summary}",
            f"Approach: {self.approach}",
            f"Answer: {self.final_answer}",
        ]
        if self.error_analysis:
            parts.append(f"Error: {self.error_analysis}")
        return "\n".join(parts)


class DuplicateCheckResult(BaseModel):
    """Result of duplicate memory check."""
    decision: str  # "STORE" | "SKIP"
    reason: str


class TaskMemoryManager:
    """Manages long-term task memories using ChromaDB."""

    def __init__(self, llm_client: "LlmClient", collection_name: str = "task_memories"):
        self.llm_client = llm_client
        self.collection_name = collection_name
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name
            )
        return self._collection

    async def save(self, context: ExecutionContext) -> str | None:
        """Extract and save a memory from the execution context."""
        execution_history = self._format_execution_history(context.events)
        memory = await self._extract_memory(execution_history)

        if memory is None:
            return None

        # Check for duplicates
        existing = await self.search(memory.task_summary, top_k=3)
        if existing and await self._is_duplicate(memory, existing):
            return None

        # Store in ChromaDB
        memory_id = str(uuid.uuid4())
        self.collection.add(
            ids=[memory_id],
            documents=[memory.to_embedding_text()],
            metadatas=[memory.model_dump()],
        )
        return memory_id

    async def search(self, query: str, top_k: int = 5) -> list[TaskMemory]:
        """Search for relevant memories."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
        )

        memories = []
        if results["metadatas"]:
            for metadata in results["metadatas"][0]:
                memories.append(TaskMemory.model_validate(metadata))
        return memories

    async def _extract_memory(self, execution_history: str) -> TaskMemory | None:
        """Extract a TaskMemory from execution history using LLM."""
        from scratch_agent.llm import LlmRequest

        prompt = f"""Analyze this agent execution and extract a memory.

Execution History:
{execution_history}

Extract:
- task_summary: Brief description of the task
- approach: How the agent solved it
- final_answer: The final answer given
"""
        request = LlmRequest(
            instructions=[prompt],
            contents=[Message(role="user", content="Extract the memory.")],
        )

        response = await self.llm_client.generate(request)
        for item in response.content:
            if isinstance(item, Message):
                try:
                    data = json.loads(item.content)
                    return TaskMemory.model_validate(data)
                except (json.JSONDecodeError, Exception):
                    return TaskMemory(
                        task_summary=item.content[:200],
                        approach="",
                        final_answer="",
                    )
        return None

    async def _is_duplicate(self, new_memory: TaskMemory, existing_results: list[TaskMemory]) -> bool:
        """Check if a memory is a duplicate of existing ones."""
        from scratch_agent.llm import LlmRequest

        existing_text = "\n".join(m.to_embedding_text() for m in existing_results[:3])
        prompt = f"""Compare this new memory with existing ones and decide if it's a duplicate.

New memory:
{new_memory.to_embedding_text()}

Existing memories:
{existing_text}

Respond with JSON: {{"decision": "STORE" or "SKIP", "reason": "..."}}"""

        request = LlmRequest(
            instructions=[prompt],
            contents=[Message(role="user", content="Is this a duplicate?")],
        )

        response = await self.llm_client.generate(request)
        for item in response.content:
            if isinstance(item, Message):
                try:
                    result = DuplicateCheckResult.model_validate_json(item.content)
                    return result.decision == "SKIP"
                except Exception:
                    return False
        return False

    def _format_execution_history(self, events: list[Event]) -> str:
        """Format events into readable text."""
        lines = []
        for event in events:
            for item in event.content:
                if isinstance(item, Message):
                    lines.append(f"[{item.role}]: {item.content[:500]}")
                elif isinstance(item, ToolCall):
                    lines.append(f"[Tool Call]: {item.name}({item.arguments})")
                elif isinstance(item, ToolResult):
                    content_preview = str(item.content[0])[:200] if item.content else ""
                    lines.append(f"[Tool Result]: {item.name} -> {content_preview}")
        return "\n".join(lines)
