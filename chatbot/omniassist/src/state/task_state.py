"""State schema for the Task Management subgraph."""

from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from src.state import append_reducer


class TaskState(TypedDict):
    messages: Annotated[list, add_messages]
    task_action: str
    task_data: dict
    tasks: Annotated[list[dict], append_reducer]
    affected_count: int
    related_tasks: list[dict]
    confirmation_required: bool
    user_id: str
