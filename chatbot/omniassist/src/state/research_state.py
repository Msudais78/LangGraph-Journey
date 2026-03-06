"""State schema for the Research subgraph."""

from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from src.state import append_reducer


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    search_queries: list[str]
    web_results: Annotated[list[dict], append_reducer]
    kb_results: Annotated[list[dict], append_reducer]
    academic_results: Annotated[list[dict], append_reducer]
    validated_sources: Annotated[list[dict], append_reducer]
    synthesis: str
    quality_score: float
    needs_more_research: bool
    user_id: str
