"""State schema for the Data Analysis subgraph."""

from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from src.state import append_reducer


class DataState(TypedDict):
    messages: Annotated[list, add_messages]
    data_source: str
    raw_data: str
    analysis_type: str
    analysis_results: Annotated[list[dict], append_reducer]
    visualizations: Annotated[list[str], append_reducer]
    summary: str
    user_id: str
