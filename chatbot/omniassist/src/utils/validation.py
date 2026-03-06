"""Input validation utilities."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class MessageInput(BaseModel):
    """Validate a single message input."""
    content: str
    role: str = "human"

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        valid_roles = {"human", "ai", "system", "tool"}
        if v.lower() not in valid_roles:
            raise ValueError(f"Invalid role: {v}. Must be one of {valid_roles}")
        return v.lower()


def validate_thread_id(thread_id: str) -> str:
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id cannot be empty")
    return thread_id.strip()


def validate_user_id(user_id: str) -> str:
    if not user_id or not user_id.strip():
        raise ValueError("user_id cannot be empty")
    return user_id.strip()
