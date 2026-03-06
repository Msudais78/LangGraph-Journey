"""Tools that directly update graph state using Command."""

from langchain_core.tools import tool
from langgraph.types import Command


@tool
def set_user_preference(preference_key: str, preference_value: str) -> Command:
    """Set a user preference that persists in the conversation state.

    Args:
        preference_key: The preference name (e.g. 'theme', 'language')
        preference_value: The preference value (e.g. 'dark', 'spanish')
    """
    return Command(update={
        "metadata": {preference_key: preference_value},
        "tool_results": [{"tool": "set_user_preference", "key": preference_key, "value": preference_value}]
    })


@tool
def flag_for_review(reason: str) -> Command:
    """Flag the current conversation for human review.

    Args:
        reason: Why this conversation needs human review
    """
    return Command(update={
        "requires_human_approval": True,
        "tool_results": [{"tool": "flag_for_review", "reason": reason}]
    })
