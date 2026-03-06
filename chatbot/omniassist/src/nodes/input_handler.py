"""Input handler node — validates and preprocesses incoming messages."""

from __future__ import annotations

from src.state.main_state import ValidatedInput


def input_handler_node(state: dict) -> dict:
    """Validate and preprocess user input."""
    try:
        validated = ValidatedInput(
            messages=state.get("messages", []),
            user_id=state.get("user_id", "default_user"),
        )
    except Exception as e:
        return {
            "error_log": [f"Input validation error: {str(e)}"],
            "current_agent": "error_handler",
        }

    return {
        "user_id": validated.user_id,
        "retry_count": 0,
    }
