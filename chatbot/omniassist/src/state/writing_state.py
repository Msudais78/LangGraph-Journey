"""State schema for the Writing subgraph."""

from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from src.state import append_reducer


class WritingState(TypedDict):
    messages: Annotated[list, add_messages]
    writing_request: str
    content_type: str
    draft_content: str
    revision_history: Annotated[list[str], append_reducer]
    quality_score: float
    human_feedback: str | None
    action: str
    iteration_count: int
    user_id: str
