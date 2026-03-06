"""Agent setup wrappers for supervisor, swarm, and bigtool patterns."""

from __future__ import annotations


def create_omniassist_supervisor(agents: list, model: str = "llama-3.3-70b-versatile"):
    """Create a supervisor that delegates to specialist agents."""
    try:
        from langgraph_supervisor import create_supervisor
        return create_supervisor(agents=agents, model=model)
    except ImportError:
        return None


def create_omniassist_swarm(agents: list):
    """Create a swarm of agents that can hand off to each other."""
    try:
        from langgraph_swarm import create_swarm
        return create_swarm(agents=agents)
    except ImportError:
        return None


def create_bigtool_agent(tools: list, model: str = "llama-3.3-70b-versatile"):
    """Create an agent optimized for many tools using langgraph-bigtool."""
    try:
        from langgraph_bigtool import create_agent_with_many_tools
        return create_agent_with_many_tools(model=model, tools=tools)
    except ImportError:
        return None
