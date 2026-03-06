"""Calendar tool (mock implementation)."""

from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def schedule_event(title: str, date: str, time: str) -> str:
    """Schedule a calendar event.

    Args:
        title: Event title
        date: Event date in YYYY-MM-DD format
        time: Event time in HH:MM format
    """
    return f"Event '{title}' scheduled for {date} at {time}."
