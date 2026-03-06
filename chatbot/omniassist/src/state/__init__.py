"""Custom reducers for state channels."""

from typing import Any


def append_reducer(existing: list | None, new: list | Any) -> list:
    """Reducer that appends new items to existing list.

    If `new` is a list, extend. If single item, append.
    """
    if existing is None:
        existing = []
    if isinstance(new, list):
        return existing + new
    return existing + [new]


def replace_reducer(existing: Any, new: Any) -> Any:
    """Reducer that simply replaces the old value with the new one."""
    return new
