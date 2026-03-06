"""MCP tool server connections."""

from __future__ import annotations


MCP_SERVER_CONFIGS = {
    "file_system": {
        "url": "http://localhost:8080/mcp",
        "transport": "streamable_http",
    },
}


def get_mcp_server_config(server_name: str) -> dict | None:
    """Get MCP server config by name."""
    return MCP_SERVER_CONFIGS.get(server_name)
