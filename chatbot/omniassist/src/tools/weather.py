"""Weather tool (mock implementation)."""

from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: City name or location, e.g. 'San Francisco, CA'
    """
    return f"Weather in {location}: 72°F, partly cloudy, humidity 55%."
