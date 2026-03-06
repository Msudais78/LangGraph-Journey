"""Assistant configurations — different personas/configurations for the same graph."""

from __future__ import annotations


ASSISTANT_CONFIGS = {
    "default": {
        "model_name": "llama-3.3-70b-versatile",
        "temperature": 0.7,
        "persona": "professional",
        "max_research_sources": 5,
        "code_execution_enabled": True,
        "language": "en",
    },
    "creative": {
        "model_name": "llama-3.3-70b-versatile",
        "temperature": 1.0,
        "persona": "casual",
        "max_research_sources": 3,
        "code_execution_enabled": True,
        "language": "en",
    },
    "fast": {
        "model_name": "llama-3.1-8b-instant",
        "temperature": 0.5,
        "persona": "professional",
        "max_research_sources": 3,
        "code_execution_enabled": True,
        "language": "en",
    },
    "code_assistant": {
        "model_name": "llama-3.3-70b-versatile",
        "temperature": 0.1,
        "persona": "technical",
        "max_research_sources": 2,
        "code_execution_enabled": True,
        "language": "en",
    },
}


def get_assistant_config(name: str = "default") -> dict:
    """Get assistant configuration by name."""
    return ASSISTANT_CONFIGS.get(name, ASSISTANT_CONFIGS["default"])
