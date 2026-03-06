"""Tests for streaming utilities."""

from src.utils.streaming import progress_event, thinking_event, agent_handoff_event


def test_progress_event():
    event = progress_event("searching", 50, "Looking up sources")
    assert event["type"] == "progress"
    assert event["data"]["percent"] == 50


def test_thinking_event():
    event = thinking_event("Analyzing...")
    assert event["type"] == "thinking"


def test_agent_handoff_event():
    event = agent_handoff_event("supervisor", "research")
    assert event["data"]["from"] == "supervisor"
    assert event["data"]["to"] == "research"
