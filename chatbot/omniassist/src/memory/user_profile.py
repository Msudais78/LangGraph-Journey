"""User profile management — persisted across threads via Store."""

from __future__ import annotations

from langgraph.store.base import BaseStore


def get_user_profile(store: BaseStore, user_id: str) -> dict:
    """Retrieve user profile from the store."""
    try:
        results = store.search(("users", user_id, "profile"), query="", limit=10)
        profile = {}
        for r in results:
            profile.update(r.value)
        return profile
    except Exception:
        return {}


def update_user_profile(store: BaseStore, user_id: str, updates: dict):
    """Update user profile in the store."""
    try:
        for key, value in updates.items():
            store.put(("users", user_id, "profile"), key, {"text": str(value), key: value})
    except Exception:
        pass
