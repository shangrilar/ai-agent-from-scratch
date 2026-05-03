"""Context optimization strategies: sliding window, compaction, summarization."""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from scratch_agents.context import ExecutionContext
from scratch_agents.types import ContentItem, Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from scratch_agents.llm import LlmClient, LlmRequest, LlmResponse


def create_optimizer_callback(apply_optimization, threshold: int = 50000):
    """Factory function that creates a callback applying optimization strategy."""

    async def callback(
        context: ExecutionContext,
        request: "LlmRequest",
    ) -> Optional["LlmResponse"]:
        token_count = count_tokens(request)

        if token_count < threshold:
            return None

        # Support both sync and async functions
        result = apply_optimization(context, request)
        if inspect.isawaitable(result):
            await result
        return None

    return callback


def count_tokens(request: "LlmRequest") -> int:
    """Calculate total token count of LlmRequest."""
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    # Build messages to count tokens from actual API format
    from scratch_agents.llm import LlmClient
    client = LlmClient(model="gpt-4")
    messages = client._build_messages(request)
    total_tokens = 0

    for message in messages:
        total_tokens += 4  # per-message overhead

        if message.get("content"):
            total_tokens += len(encoding.encode(str(message["content"])))

        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                func = tool_call.get("function", {})
                if func.get("name"):
                    total_tokens += len(encoding.encode(func["name"]))
                if func.get("arguments"):
                    total_tokens += len(encoding.encode(func["arguments"]))

    # Tool definition tokens
    if request.tools:
        for tool in request.tools:
            tool_def = tool.tool_definition
            if tool_def:
                total_tokens += len(encoding.encode(json.dumps(tool_def)))

    return total_tokens


def apply_sliding_window(
    context: ExecutionContext,
    request: "LlmRequest",
    window_size: int = 20,
) -> None:
    """Sliding window that keeps only the most recent N messages."""
    contents = request.contents

    # Find user message position
    user_message_idx = None
    for i, item in enumerate(contents):
        if isinstance(item, Message) and item.role == "user":
            user_message_idx = i
            break

    if user_message_idx is None:
        return

    # Preserve up to user message
    preserved = contents[: user_message_idx + 1]

    # Keep only the most recent N from remaining items
    remaining = contents[user_message_idx + 1 :]
    if len(remaining) > window_size:
        remaining = remaining[-window_size:]

    request.contents = preserved + remaining


# Tools to compress ToolCall arguments
TOOLCALL_COMPACTION_RULES = {
    "create_file": "[Content saved to file]",
}

# Tools to compress ToolResult content
TOOLRESULT_COMPACTION_RULES = {
    "read_file": "File content from {file_path}. Re-read if needed.",
    "search_web": "Search results processed. Query: {query}. Re-search if needed.",
    "tavily_search": "Search results processed. Query: {query}. Re-search if needed.",
}


def apply_compaction(context: ExecutionContext, request: "LlmRequest") -> None:
    """Compress tool calls and results into reference messages."""
    tool_call_args: Dict[str, Dict] = {}
    compacted = []

    for item in request.contents:
        if isinstance(item, ToolCall):
            tool_call_args[item.tool_call_id] = item.arguments

            if item.name in TOOLCALL_COMPACTION_RULES:
                compressed_args = {
                    k: TOOLCALL_COMPACTION_RULES[item.name] if k == "content" else v
                    for k, v in item.arguments.items()
                }
                compacted.append(
                    ToolCall(
                        tool_call_id=item.tool_call_id,
                        name=item.name,
                        arguments=compressed_args,
                    )
                )
            else:
                compacted.append(item)

        elif isinstance(item, ToolResult):
            if item.name in TOOLRESULT_COMPACTION_RULES:
                args = tool_call_args.get(item.tool_call_id, {})
                template = TOOLRESULT_COMPACTION_RULES[item.name]
                compressed_content = template.format(
                    file_path=args.get("file_path", args.get("path", "unknown")),
                    query=args.get("query", "unknown"),
                )
                compacted.append(
                    ToolResult(
                        tool_call_id=item.tool_call_id,
                        name=item.name,
                        status=item.status,
                        content=[compressed_content],
                    )
                )
            else:
                compacted.append(item)
        else:
            compacted.append(item)

    request.contents = compacted


SUMMARIZATION_PROMPT = """You are summarizing an AI agent's work progress.

Given the following execution history, extract:
1. Key findings: Important information discovered
2. Tools used: List of tools that were called
3. Current status: What has been accomplished and what remains

Be concise. Focus on information that will help the agent continue its work.

Execution History:
{history}

Provide a structured summary."""


async def apply_summarization(
    context: ExecutionContext,
    request: "LlmRequest",
    llm_client: "LlmClient",
    keep_recent: int = 5,
) -> None:
    """Replace old messages with a summary."""
    from scratch_agents.llm import LlmRequest as LR

    contents = request.contents

    # Find user message position
    user_idx = None
    for i, item in enumerate(contents):
        if isinstance(item, Message) and item.role == "user":
            user_idx = i
            break

    if user_idx is None:
        return

    # Check previous summary position
    last_summary_idx = context.state.get("last_summary_idx", user_idx)

    # Calculate summarization target range
    summary_start = last_summary_idx + 1
    summary_end = len(contents) - keep_recent

    if summary_end <= summary_start:
        return

    preserved_start = contents[: last_summary_idx + 1]
    preserved_end = contents[summary_end:]
    to_summarize = contents[summary_start:summary_end]

    # Generate summary
    history_text = _format_history_for_summary(to_summarize)
    summary = await _generate_summary(llm_client, history_text)

    # Add summary to instructions
    request.instructions.append(f"[Previous work summary]\n{summary}")

    # Keep only preserved portions
    request.contents = preserved_start + preserved_end

    # Record summary position
    context.state["last_summary_idx"] = len(preserved_start) - 1


def _format_history_for_summary(items: List[ContentItem]) -> str:
    """Convert ContentItem list to text for summarization."""
    lines = []
    for item in items:
        if isinstance(item, Message):
            lines.append(f"[{item.role}]: {item.content[:500]}...")
        elif isinstance(item, ToolCall):
            lines.append(f"[Tool Call]: {item.name}({item.arguments})")
        elif isinstance(item, ToolResult):
            content_preview = str(item.content[0])[:200] if item.content else ""
            lines.append(f"[Tool Result]: {item.name} -> {content_preview}...")
    return "\n".join(lines)


async def _generate_summary(llm_client: "LlmClient", history: str) -> str:
    """Generate history summary using LLM."""
    from scratch_agents.llm import LlmRequest as LR

    request = LR(
        instructions=[SUMMARIZATION_PROMPT.format(history=history)],
        contents=[Message(role="user", content="Please summarize.")],
    )

    response = await llm_client.generate(request)

    for item in response.content:
        if isinstance(item, Message):
            return item.content

    return ""


class ContextOptimizer:
    """Hierarchical context optimization strategy."""

    def __init__(
        self,
        llm_client: "LlmClient",
        token_threshold: int = 50000,
        enable_compaction: bool = True,
        enable_summarization: bool = True,
        keep_recent: int = 5,
    ):
        self.llm_client = llm_client
        self.token_threshold = token_threshold
        self.enable_compaction = enable_compaction
        self.enable_summarization = enable_summarization
        self.keep_recent = keep_recent

    async def __call__(
        self,
        context: ExecutionContext,
        request: "LlmRequest",
    ) -> Optional["LlmResponse"]:
        """Register as before_llm_callback."""
        if count_tokens(request) < self.token_threshold:
            return None

        if self.enable_compaction:
            apply_compaction(context, request)
            if count_tokens(request) < self.token_threshold:
                return None

        if self.enable_summarization:
            await apply_summarization(
                context, request, self.llm_client, self.keep_recent
            )

        return None
