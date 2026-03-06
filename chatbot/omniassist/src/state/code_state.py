"""State schema for the Code Execution module."""

from __future__ import annotations
from typing_extensions import TypedDict


class CodeState(TypedDict):
    request: str
    generated_code: str
    language: str
    safety_level: str
    review_notes: str
    execution_output: str
    execution_error: str | None
    approved: bool
    user_id: str
