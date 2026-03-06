"""Tests for memory system."""

from src.memory.long_term import create_memory_store, store_user_fact, search_user_facts


def test_create_memory_store():
    store = create_memory_store()
    assert store is not None


def test_store_and_retrieve_fact():
    store = create_memory_store()
    store_user_fact(store, "user1", "fact1", {"text": "User likes Python"})
    facts = search_user_facts(store, "user1", "Python")
    assert isinstance(facts, list)
