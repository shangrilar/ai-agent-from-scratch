"""Tool callbacks for the agent: approval and compression."""

from scratch_agent.rag import fixed_length_chunking, get_embeddings, vector_search
from scratch_agent.context import ExecutionContext
from scratch_agent.types import ToolCall, ToolResult, Message


DANGEROUS_TOOLS = ["delete_file", "send_email", "execute_sql"]


def approval_callback(context: ExecutionContext, tool_call: ToolCall):
    """Requests user approval before executing dangerous tools."""
    if tool_call.name not in DANGEROUS_TOOLS:
        return None

    print(f"\n⚠️ Dangerous tool execution requested")
    print(f"Tool: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")

    response = input("Do you want to execute? (y/n): ").lower().strip()

    if response == 'y':
        print("✅ Approved. Executing...\n")
        return None
    else:
        print("❌ Denied. Skipping execution.\n")
        return f"User denied execution of {tool_call.name}"


def search_compressor(context: ExecutionContext, tool_result: ToolResult):
    """Callback that compresses web search results."""
    if tool_result.name != "search_web":
        return None

    original_content = tool_result.content[0]

    if len(original_content) < 2000:
        return None

    query = _extract_search_query(context, tool_result.tool_call_id)
    if not query:
        return None

    chunks = fixed_length_chunking(original_content, chunk_size=500, overlap=50)
    embeddings = get_embeddings(chunks)
    results = vector_search(query, chunks, embeddings, top_k=3)

    compressed = "\n\n".join([r['chunk'] for r in results])

    return ToolResult(
        tool_call_id=tool_result.tool_call_id,
        name=tool_result.name,
        status="success",
        content=[compressed]
    )


def _extract_search_query(context: ExecutionContext, tool_call_id: str) -> str:
    """Extract the original search query from context."""
    for event in context.events:
        for item in event.content:
            if isinstance(item, ToolCall) and item.name == "search_web" and item.tool_call_id == tool_call_id:
                return item.arguments.get("query", "")
    return ""
