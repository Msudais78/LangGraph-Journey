"""Output handler node — formats final response."""

from __future__ import annotations


def output_handler_node(state: dict) -> dict:
    """Final node that prepares the output."""
    return {
        "current_agent": state.get("current_agent", "unknown"),
    }
