"""Core state schema for OmniAssist main orchestrator graph."""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, field_validator

from src.state import append_reducer


class OmniAssistState(TypedDict):
    """Full internal state for the main orchestrator."""

    # Conversation
    messages: Annotated[list, add_messages]
    current_intent: str
    conversation_summary: str

    # User
    user_id: str
    thread_id: str

    # Agent outputs
    tool_results: Annotated[list[dict], append_reducer]
    research_results: Annotated[list[dict], append_reducer]
    draft_content: str
    tasks: Annotated[list[dict], append_reducer]
    code_output: str
    data_analysis: dict

    # Control flow
    current_agent: str
    requires_human_approval: bool
    human_feedback: str | None
    error_log: Annotated[list[str], append_reducer]
    retry_count: int
    metadata: dict


class OmniAssistInput(TypedDict):
    """Public input schema — only messages accepted."""
    messages: list


class OmniAssistOutput(TypedDict):
    """Public output schema — only messages and current_agent returned."""
    messages: list
    current_agent: str


class ValidatedInput(BaseModel):
    """Pydantic model used to validate incoming user input at graph boundary."""
    messages: list
    user_id: str = "default_user"

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_id cannot be empty")
        return v.strip()
