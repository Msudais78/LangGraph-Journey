"""Checkpointer factory — InMemory, SQLite, or PostgreSQL backends."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver


def get_checkpointer(backend: str = "memory", connection_string: str = "", encrypted: bool = False):
    """Factory for checkpointer backends.

    Args:
        backend: "memory", "sqlite", or "postgres"
        connection_string: DB connection string (for sqlite/postgres)
        encrypted: Whether to encrypt checkpoint data (requires pycryptodome)
    """
    if backend == "memory":
        return InMemorySaver()

    elif backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn_str = connection_string or "omniassist.db"
            return SqliteSaver.from_conn_string(conn_str)
        except ImportError:
            print("langgraph-checkpoint-sqlite not installed. Falling back to memory.")
            return InMemorySaver()

    elif backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            if not connection_string:
                raise ValueError("PostgresSaver requires a connection string")
            checkpointer = PostgresSaver.from_conn_string(connection_string)
            checkpointer.setup()
            return checkpointer
        except ImportError:
            print("langgraph-checkpoint-postgres not installed. Falling back to memory.")
            return InMemorySaver()

    else:
        raise ValueError(f"Unknown backend: {backend}")
