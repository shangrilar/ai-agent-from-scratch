from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from scratch_agent.tools.helpers import format_tool_definition


async def load_mcp_tools(connection: dict):
    """Load tools from an MCP server connection.

    Args:
        connection: Dict with 'command' and 'args' for the MCP server

    Returns:
        ListToolsResult with a `.tools` attribute (matches Listing 3.22 / 3.23).
    """
    server_params = StdioServerParameters(
        command=connection["command"],
        args=connection.get("args", []),
        env=connection.get("env"),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.list_tools()


def mcp_tools_to_openai_format(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to OpenAI tool format."""
    return [
        format_tool_definition(
            name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema,
        )
        for tool in mcp_tools.tools
    ]


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
