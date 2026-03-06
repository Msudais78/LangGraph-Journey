"""Long-term memory — cross-thread persistence using Store API."""

from __future__ import annotations

from langgraph.store.memory import InMemoryStore


def create_memory_store(use_semantic_search: bool = False) -> InMemoryStore:
    """Create an in-memory store.

    For production, replace with PostgresStore.
    Semantic search requires an embedding provider — omitted here since we use Groq (no embeddings).
    """
    return InMemoryStore()


def store_user_fact(store, user_id: str, fact_id: str, fact: dict):
    """Store a user fact in long-term memory."""
    store.put(("users", user_id, "facts"), fact_id, fact)


def search_user_facts(store, user_id: str, query: str, limit: int = 5) -> list:
    """Search user facts."""
    try:
        results = store.search(("users", user_id, "facts"), query=query, limit=limit)
        return [{"key": r.key, "value": r.value} for r in results]
    except Exception:
        return []


def store_user_preference(store, user_id: str, key: str, value: str):
    """Store a user preference."""
    store.put(("users", user_id, "preferences"), key, {"text": value, "key": key})


def get_user_preferences(store, user_id: str) -> dict:
    """Get all user preferences."""
    try:
        results = store.search(("users", user_id, "preferences"), query="", limit=50)
        return {r.value.get("key", r.key): r.value for r in results}
    except Exception:
        return {}
