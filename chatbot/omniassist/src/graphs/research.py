"""Research Subgraph — performs parallel web search, knowledge base search,
and academic search, then validates and synthesizes results.

Concepts: Fan-out/Fan-in, deferred nodes, quality loop cycle, retry policy.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from src.config.models import get_chat_model
from src.config.prompts import RESEARCH_AGENT_PROMPT
from src.state import append_reducer
from src.state.research_state import ResearchState
from src.tools.web_search import web_search
from src.tools.knowledge_base import search_knowledge_base


def query_planner_node(state: ResearchState) -> dict:
    """Plan search queries based on user's research request."""
    model = get_chat_model(temperature=0)
    last_message = state["messages"][-1].content if state["messages"] else state.get("query", "")

    prompt = [
        SystemMessage(content=(
            "Generate 2-3 specific search queries to research this topic thoroughly. "
            "Return each query on a new line, nothing else."
        )),
        HumanMessage(content=last_message),
    ]
    response = model.invoke(prompt)
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]

    return {"search_queries": queries, "query": last_message}


def web_search_node(state: ResearchState) -> dict:
    """Execute web searches for all planned queries."""
    results = []
    for query in state.get("search_queries", [])[:3]:
        try:
            search_result = web_search.invoke({"query": query})
            results.append({"source": "web", "query": query, "content": search_result, "title": query})
        except Exception as e:
            results.append({"source": "web", "query": query, "content": f"Error: {e}", "title": query})
    return {"web_results": results}


def knowledge_base_node(state: ResearchState) -> dict:
    """Search internal knowledge base."""
    query = state.get("query", "")
    try:
        result = search_knowledge_base.invoke({"query": query})
        return {"kb_results": [{"source": "knowledge_base", "query": query, "content": result, "title": f"KB: {query}"}]}
    except Exception as e:
        return {"kb_results": [{"source": "knowledge_base", "query": query, "content": f"Error: {e}", "title": query}]}


def academic_search_node(state: ResearchState) -> dict:
    """Search academic sources (mock)."""
    query = state.get("query", "")
    return {
        "academic_results": [{
            "source": "academic",
            "query": query,
            "content": f"Academic findings on '{query}': [Mock academic search result]",
            "title": f"Academic: {query}",
        }]
    }


def results_aggregator_node(state: ResearchState) -> dict:
    """Aggregate all search results from parallel branches. (DEFERRED node)"""
    all_results = (
        state.get("web_results", [])
        + state.get("kb_results", [])
        + state.get("academic_results", [])
    )
    return {"validated_sources": all_results}


def synthesis_node(state: ResearchState) -> dict:
    """Synthesize validated research into a coherent response."""
    model = get_chat_model(temperature=0.3)

    sources = state.get("validated_sources", [])
    source_texts = []
    for i, src in enumerate(sources, 1):
        source_texts.append(f"[{i}] ({src.get('source', 'unknown')}): {src.get('content', '')[:300]}")

    prompt = [
        SystemMessage(content=RESEARCH_AGENT_PROMPT),
        HumanMessage(content=(
            f"Research query: {state.get('query', '')}\n\n"
            f"Sources found:\n" + "\n".join(source_texts) + "\n\n"
            "Synthesize these into a clear, well-organized response. "
            "Cite sources by number [1], [2], etc."
        )),
    ]

    response = model.invoke(prompt)
    return {
        "synthesis": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def quality_check_node(state: ResearchState) -> dict:
    """Check the quality of the synthesis."""
    model = get_chat_model(temperature=0)
    synthesis = state.get("synthesis", "")

    prompt = [
        SystemMessage(content="Rate this research synthesis quality 1-10. Respond with just the number."),
        HumanMessage(content=synthesis[:1000]),
    ]

    try:
        response = model.invoke(prompt)
        score = float(response.content.strip().split()[0])
    except Exception:
        score = 7.0  # Default pass score

    return {"quality_score": score, "needs_more_research": score < 6.0}


def should_research_more(state: ResearchState) -> str:
    """Conditional edge: should we loop back for more research?"""
    if state.get("needs_more_research", False):
        return "needs_improvement"
    return "good_enough"


def build_research_graph() -> StateGraph:
    """Build the research subgraph with parallel search and quality loop."""
    builder = StateGraph(ResearchState)

    builder.add_node("query_planner", query_planner_node)
    builder.add_node(
        "web_search", web_search_node,
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0)
    )
    builder.add_node("knowledge_base", knowledge_base_node)
    builder.add_node("academic_search", academic_search_node)
    builder.add_node("results_aggregator", results_aggregator_node, defer=True)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("quality_check", quality_check_node)

    builder.add_edge(START, "query_planner")

    # Fan-out: parallel search branches
    builder.add_edge("query_planner", "web_search")
    builder.add_edge("query_planner", "knowledge_base")
    builder.add_edge("query_planner", "academic_search")

    # Fan-in: all branches feed into deferred aggregator
    builder.add_edge("web_search", "results_aggregator")
    builder.add_edge("knowledge_base", "results_aggregator")
    builder.add_edge("academic_search", "results_aggregator")

    builder.add_edge("results_aggregator", "synthesis")
    builder.add_edge("synthesis", "quality_check")

    builder.add_conditional_edges(
        "quality_check",
        should_research_more,
        {
            "needs_improvement": "query_planner",
            "good_enough": END,
        }
    )

    return builder


graph = build_research_graph().compile()
