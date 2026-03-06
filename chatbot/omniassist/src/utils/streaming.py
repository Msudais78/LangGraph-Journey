"""Streaming utilities — stream event helpers."""

from __future__ import annotations

from typing import Any


def format_stream_event(event_type: str, data: Any) -> dict:
    """Create a standardized stream event dictionary."""
    return {"type": event_type, "data": data}


def progress_event(step: str, percent: int, detail: str = "") -> dict:
    """Create a progress stream event."""
    return format_stream_event("progress", {"step": step, "percent": percent, "detail": detail})


def thinking_event(content: str) -> dict:
    """Create a 'thinking' stream event for UI."""
    return format_stream_event("thinking", {"content": content})


def agent_handoff_event(from_agent: str, to_agent: str) -> dict:
    """Create an agent handoff notification event."""
    return format_stream_event("agent_handoff", {"from": from_agent, "to": to_agent})
