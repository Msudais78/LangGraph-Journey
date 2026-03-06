"""Memory system — short-term message trimming."""

from __future__ import annotations


def trim_conversation(messages: list, max_messages: int = 20) -> list:
    """Trim a conversation to keep only the most recent messages."""
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]
