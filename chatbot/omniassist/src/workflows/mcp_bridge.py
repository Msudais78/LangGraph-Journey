"""MCP Bridge — connects to MCP-compatible tool servers."""

from __future__ import annotations


async def get_mcp_tools(servers: dict | None = None):
    """Get tools from MCP servers."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        return []

    if servers is None:
        servers = {
            "file_system": {
                "url": "http://localhost:8080/mcp",
                "transport": "streamable_http",
            },
        }

    try:
        async with MultiServerMCPClient(servers) as client:
            return client.get_tools()
    except Exception as e:
        print(f"MCP connection error: {e}")
        return []
