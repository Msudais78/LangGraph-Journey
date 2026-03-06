"""Global error handler node."""

from __future__ import annotations

from langchain_core.messages import AIMessage


def error_handler_node(state: dict) -> dict:
    """Handle errors gracefully."""
    errors = state.get("error_log", [])
    last_error = errors[-1] if errors else "Unknown error occurred"

    error_message = AIMessage(
        content=f"I apologize, but I encountered an issue: {last_error}. "
                f"Please try again or rephrase your request."
    )

    return {
        "messages": [error_message],
        "current_agent": "error_handler",
    }
