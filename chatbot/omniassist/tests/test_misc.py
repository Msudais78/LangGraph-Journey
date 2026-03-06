"""Tests for checkpointing backends."""

from src.utils.encryption import get_checkpointer


def test_memory_checkpointer():
    cp = get_checkpointer("memory")
    assert cp is not None


def test_mcp_config():
    from src.tools.mcp_tools import get_mcp_server_config, MCP_SERVER_CONFIGS
    assert isinstance(MCP_SERVER_CONFIGS, dict)
    config = get_mcp_server_config("file_system")
    if config:
        assert "transport" in config


def test_functional_api_imports():
    from src.workflows.code_pipeline import code_pipeline, generate_code, execute_code
    assert code_pipeline is not None

    from src.workflows.quick_research import quick_research
    assert quick_research is not None
