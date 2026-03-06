"""Quick Research — lightweight research using Functional API."""

from __future__ import annotations

from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

from src.config.models import get_chat_model
from src.tools.web_search import web_search


@task(retry_policy=RetryPolicy(max_attempts=2))
def quick_search(query: str) -> str:
    """Perform a quick web search."""
    try:
        return web_search.invoke({"query": query})
    except Exception as e:
        return f"Search failed: {e}"


@task
def quick_summarize(query: str, search_results: str) -> str:
    """Summarize search results concisely."""
    model = get_chat_model(temperature=0.3)
    from langchain_core.messages import SystemMessage, HumanMessage

    response = model.invoke([
        SystemMessage(content="Summarize these search results concisely in 2-3 sentences."),
        HumanMessage(content=f"Query: {query}\n\nResults:\n{search_results}"),
    ])
    return response.content


@entrypoint(checkpointer=InMemorySaver())
def quick_research(query: str) -> dict:
    """Quick research pipeline — search and summarize in one shot."""
    results = quick_search(query).result()
    summary = quick_summarize(query, results).result()
    return {"query": query, "results": results, "summary": summary}
