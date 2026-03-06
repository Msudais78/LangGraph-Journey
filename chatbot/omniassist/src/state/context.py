"""Context schema for OmniAssist."""

from typing_extensions import TypedDict


class OmniAssistContext(TypedDict, total=False):
    """Run-scoped context injected into the graph at invocation time."""
    model_name: str
    temperature: float
    persona: str
    max_research_sources: int
    code_execution_enabled: bool
    language: str
