"""Source Validation — nested subgraph used within the Research subgraph."""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from src.config.models import get_chat_model
from src.state import append_reducer


class SourceValidationState(TypedDict):
    sources: list[dict]
    validated_sources: Annotated[list[dict], append_reducer]
    rejected_sources: Annotated[list[dict], append_reducer]


def validate_source_node(state: SourceValidationState) -> dict:
    """Validate each source for reliability and relevance."""
    model = get_chat_model(temperature=0)
    validated = []
    rejected = []

    for source in state.get("sources", []):
        prompt = [
            SystemMessage(content="Rate this source reliability 1-10. Respond with just the number."),
            HumanMessage(content=f"Source: {source.get('title', 'Unknown')} - {source.get('content', '')[:300]}"),
        ]
        try:
            response = model.invoke(prompt)
            score = int(response.content.strip().split()[0])
            source_with_score = {**source, "reliability_score": score}
            if score >= 5:
                validated.append(source_with_score)
            else:
                rejected.append(source_with_score)
        except Exception:
            source_with_score = {**source, "reliability_score": 0}
            rejected.append(source_with_score)

    return {"validated_sources": validated, "rejected_sources": rejected}


def check_enough_sources(state: SourceValidationState) -> str:
    """Conditional edge: check if we have enough validated sources."""
    if len(state.get("validated_sources", [])) >= 2:
        return END
    return "needs_more"


def build_source_validator() -> StateGraph:
    """Build the source validation nested subgraph."""
    builder = StateGraph(SourceValidationState)

    builder.add_node(
        "validate",
        validate_source_node,
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0, backoff_factor=2.0)
    )

    builder.add_edge(START, "validate")
    builder.add_conditional_edges("validate", check_enough_sources, {END: END, "needs_more": END})

    return builder


source_validator_graph = build_source_validator().compile()
