import json
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def load_mcp_tools(connection: dict) -> list:
    """Load tools from an MCP server connection.

    Args:
        connection: Dict with 'command' and 'args' for the MCP server

    Returns:
        List of tool objects from the MCP server
    """
    server_params = StdioServerParameters(
        command=connection["command"],
        args=connection.get("args", []),
        env=connection.get("env"),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            return tools_result.tools


def mcp_tools_to_openai_format(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function calling format."""
    openai_tools = []
    for tool in mcp_tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or f"MCP tool: {tool.name}",
                "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {"type": "object", "properties": {}},
            }
        }
        openai_tools.append(openai_tool)
    return openai_tools


@asynccontextmanager
async def mcp_connection(connection: dict):
    """Context manager for maintaining an MCP server connection.

    Usage:
        async with mcp_connection({"command": "npx", "args": [...]}) as session:
            tools = await session.list_tools()
            result = await session.call_tool("tool_name", arguments={...})
    """
    server_params = StdioServerParameters(
        command=connection["command"],
        args=connection.get("args", []),
        env=connection.get("env"),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session
