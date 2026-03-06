"""Shared human review node using interrupt() function."""

from __future__ import annotations

from langgraph.types import interrupt


def human_approval_gate(state: dict) -> dict:
    """Generic human approval gate.

    IMPORTANT: The entire node re-executes on resume. No side effects before interrupt()!
    """
    action = state.get("pending_action", "unknown action")
    details = state.get("pending_details", "")

    response = interrupt({
        "action": action,
        "details": details,
        "prompt": f"Approve '{action}'? Reply 'approved' or 'rejected'.",
    })

    return {
        "human_feedback": response,
        "requires_human_approval": False,
    }
