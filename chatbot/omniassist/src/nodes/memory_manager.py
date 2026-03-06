"""Memory manager node — handles short-term and long-term memory operations."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.store.base import BaseStore

from src.memory.short_term import trim_conversation
from src.nodes.message_utils import should_summarize


def memory_manager_node(state: dict, *, store: BaseStore) -> dict:
    """Manage conversation memory — trim and store facts.

    The `store` parameter is automatically injected by the LangGraph runtime
    when the graph is compiled with a store.
    """
    messages = state.get("messages", [])
    user_id = state.get("user_id", "default")
    updates = {}

    # Store notable facts from the latest message
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage) and len(last_msg.content) > 20:
            try:
                import uuid
                fact_id = str(uuid.uuid4())[:8]
                store.put(
                    ("users", user_id, "facts"),
                    fact_id,
                    {"text": last_msg.content[:500], "summary": last_msg.content[:100]},
                )
            except Exception:
                pass

    return updates
