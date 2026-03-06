"""LangMem SDK integration placeholder."""

from __future__ import annotations


def get_memory_manager(model: str = "llama-3.3-70b-versatile"):
    """Create a LangMem memory manager if available."""
    try:
        from langmem import create_memory_manager
        return create_memory_manager(model=model)
    except ImportError:
        return None


async def extract_memories(memory_manager, messages: list) -> list:
    """Extract key facts/memories from a conversation."""
    if memory_manager is None:
        return []
    try:
        memories = await memory_manager.ainvoke({"messages": messages})
        return memories
    except Exception:
        return []
