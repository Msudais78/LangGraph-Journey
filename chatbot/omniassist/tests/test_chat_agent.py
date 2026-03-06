"""Tests for the chat agent graph."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.graphs.chat import build_simple_chat_agent, build_custom_chat_agent, get_chat_graph


def test_custom_chat_agent_compiles():
    builder = build_custom_chat_agent()
    graph = builder.compile()
    assert graph is not None


def test_simple_chat_agent_builds():
    agent = build_simple_chat_agent()
    assert agent is not None


def test_chat_graph_with_checkpointer():
    graph = get_chat_graph(checkpointer=InMemorySaver())
    assert graph is not None
