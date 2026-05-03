"""Long-term memory with ChromaDB for task memory storage and retrieval."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pydantic import BaseModel, Field

from scratch_agents.context import ExecutionContext
from scratch_agents.types import Event, Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from scratch_agents.llm import LlmClient


class TaskMemory(BaseModel):
    """Structured memory for GAIA problem-solving records."""

    task_summary: str = Field(description="What the problem asked")
    approach: str = Field(description="Methods and tools used to solve it")
    final_answer: str = Field(description="The agent's submitted answer")
    is_correct: bool = Field(description="Whether the answer was correct")
    error_analysis: str | None = Field(
        default=None,
        description="Why the attempt failed, if it did",
    )

    def to_embedding_text(self) -> str:
        """Generate text for vector search."""
        return f"Task: {self.task_summary}"


class DuplicateCheckResult(BaseModel):
    """Result of duplicate check."""

    decision: str = Field(description="ADD (new information) or SKIP (duplicate)")
    reason: str = Field(description="Explanation for the decision")


TASK_MEMORY_EXTRACTION_PROMPT = """Analyze the following execution history and extract a structured task memory.

Execution History:
{execution_history}

Extract:
- task_summary: What the problem asked
- approach: Methods and tools used to solve it
- final_answer: The agent's submitted answer
- is_correct: Whether the answer was correct (true or false)
- error_analysis: If incorrect, explain why; otherwise leave null
"""


DUPLICATE_CHECK_PROMPT = """Compare the new memory against existing memories to determine if it's a duplicate.

Existing memories:
{existing_memories}

New memory:
{new_memory}

Respond with one of:
- ADD: This is new information that should be stored
- SKIP: Similar information already exists, no need to store

Judgment criteria:
- Same problem with different approach or different result counts as new information
- Same problem with same approach and same result is a duplicate
"""


class TaskMemoryManager:
    """Memory manager for GAIA problem-solving learning."""

    def __init__(
        self,
        llm_client: "LlmClient",
        collection_name: str = "task_memories",
    ):
        self.llm_client = llm_client

        # ChromaDB setup
        self.client = chromadb.Client()
        embedding_fn = OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )

    async def _extract_memory(self, execution_history: str) -> TaskMemory | None:
        """Extract structured memory from execution history."""
        prompt = TASK_MEMORY_EXTRACTION_PROMPT.format(
            execution_history=execution_history
        )
        try:
            return await self.llm_client.ask(
                prompt=prompt,
                response_format=TaskMemory,
            )
        except Exception as e:
            print(f"Memory extraction failed: {e}")
            return None

    def _format_execution_history(self, events: list[Event]) -> str:
        """Convert event list to text."""
        lines = []
        for event in events:
            for item in event.content:
                if isinstance(item, Message):
                    lines.append(f"[{item.role}]: {item.content}")
                elif isinstance(item, ToolCall):
                    lines.append(f"[Tool Call]: {item.name}({item.arguments})")
                elif isinstance(item, ToolResult):
                    content_preview = str(item.content[0])[:500] if item.content else ""
                    lines.append(f"[Tool Result]: {item.name} -> {content_preview}")
        return "\n".join(lines)

    async def _is_duplicate(
        self,
        new_memory: TaskMemory,
        existing_results: dict,
    ) -> bool:
        """Determine if a new memory duplicates an existing one."""
        if not existing_results["metadatas"] or not existing_results["metadatas"][0]:
            return False

        existing_texts = []
        for meta in existing_results["metadatas"][0]:
            existing_texts.append(
                f"task_summary: {meta.get('task_summary')}, "
                f"- approach: {meta.get('approach')}, "
                f"is_correct: {meta.get('is_correct')}"
            )

        prompt = DUPLICATE_CHECK_PROMPT.format(
            existing_memories="\n".join(existing_texts),
            new_memory=(
                f"task_summary: {new_memory.task_summary}, "
                f"approach: {new_memory.approach}, "
                f"is_correct: {new_memory.is_correct}"
            ),
        )

        try:
            result = await self.llm_client.ask(
                prompt=prompt,
                response_format=DuplicateCheckResult,
            )
            return result.decision == "SKIP"
        except Exception:
            return False

    async def save(self, context: ExecutionContext) -> str | None:
        """Extract and save memory from execution context.

        Returns:
            memory_id if saved, None if ignored as duplicate
        """
        # 1. Convert execution history to text
        execution_history = self._format_execution_history(context.events)

        # 2. Extract structured memory using LLM
        memory = await self._extract_memory(execution_history)
        if memory is None:
            return None

        # 3. Duplicate check using the same text used for storage
        existing = self.collection.query(
            query_texts=[memory.to_embedding_text()],
            n_results=3,
        )
        if await self._is_duplicate(memory, existing):
            return None

        # 4. Store in ChromaDB
        memory_id = str(uuid.uuid4())
        metadata = memory.model_dump()
        # ChromaDB metadata cannot store None values
        metadata = {k: ("" if v is None else v) for k, v in metadata.items()}
        self.collection.add(
            ids=[memory_id],
            documents=[memory.to_embedding_text()],
            metadatas=[metadata],
        )
        return memory_id

    async def search(self, query: str, top_k: int = 5) -> list[TaskMemory]:
        """Search for memories related to the query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        if not results["metadatas"] or not results["metadatas"][0]:
            return []

        return [TaskMemory(**meta) for meta in results["metadatas"][0]]
