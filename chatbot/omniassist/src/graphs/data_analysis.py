"""Data Analysis Subgraph — statistics, visualization, and data summary."""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from src.config.models import get_chat_model
from src.config.prompts import DATA_AGENT_PROMPT
from src.state import append_reducer
from src.state.data_state import DataState


def parse_data_request_node(state: DataState) -> dict:
    """Parse the data analysis request to determine type and data source."""
    model = get_chat_model(temperature=0)
    last_message = state["messages"][-1].content if state["messages"] else ""

    prompt = [
        SystemMessage(content=(
            "Classify this data analysis request. Respond with one word: "
            "statistics, visualization, correlation, or summary"
        )),
        HumanMessage(content=last_message),
    ]
    response = model.invoke(prompt)
    analysis_type = response.content.strip().lower().split()[0]
    valid_types = {"statistics", "visualization", "correlation", "summary"}
    if analysis_type not in valid_types:
        analysis_type = "summary"

    return {"analysis_type": analysis_type, "raw_data": last_message}


def statistics_node(state: DataState) -> dict:
    """Compute statistical analysis."""
    model = get_chat_model(temperature=0)
    data = state.get("raw_data", "")

    prompt = [
        SystemMessage(content=DATA_AGENT_PROMPT),
        HumanMessage(content=f"Provide statistical analysis of:\n{data}"),
    ]
    response = model.invoke(prompt)
    return {"analysis_results": [{"type": "statistics", "result": response.content}]}


def visualization_node(state: DataState) -> dict:
    """Generate visualization description/code."""
    model = get_chat_model(temperature=0.3)
    data = state.get("raw_data", "")

    prompt = [
        SystemMessage(content=DATA_AGENT_PROMPT),
        HumanMessage(content=f"Describe how to visualize this data (and provide simple Python code):\n{data}"),
    ]
    response = model.invoke(prompt)
    return {"analysis_results": [{"type": "visualization", "result": response.content}]}


def summary_node(state: DataState) -> dict:
    """Generate data summary."""
    model = get_chat_model(temperature=0.3)
    data = state.get("raw_data", "")

    prompt = [
        SystemMessage(content=DATA_AGENT_PROMPT),
        HumanMessage(content=f"Summarize this data analysis request and provide insights:\n{data}"),
    ]
    response = model.invoke(prompt)
    return {
        "summary": response.content,
        "analysis_results": [{"type": "summary", "result": response.content}],
    }


def results_merger_node(state: DataState) -> dict:
    """Merge all analysis results into a final response. (DEFERRED node)"""
    results = state.get("analysis_results", [])
    result_texts = []
    for r in results:
        result_texts.append(f"**{r.get('type', 'Analysis')}:**\n{r.get('result', '')}")

    combined = "\n\n".join(result_texts) if result_texts else "No analysis results available."

    return {
        "messages": [AIMessage(content=combined)],
        "summary": combined,
    }


def route_analysis_type(state: DataState) -> str:
    """Route to the appropriate analysis node."""
    analysis_type = state.get("analysis_type", "summary")
    if analysis_type == "statistics":
        return "statistics"
    elif analysis_type == "visualization":
        return "visualization"
    else:
        return "summary"


def build_data_analysis_graph() -> StateGraph:
    """Build the data analysis subgraph."""
    builder = StateGraph(DataState)

    builder.add_node("parse_request", parse_data_request_node)
    builder.add_node("statistics", statistics_node)
    builder.add_node("visualization", visualization_node)
    builder.add_node("summary", summary_node)
    builder.add_node("results_merger", results_merger_node, defer=True)

    builder.add_edge(START, "parse_request")

    builder.add_conditional_edges(
        "parse_request",
        route_analysis_type,
        {
            "statistics": "statistics",
            "visualization": "visualization",
            "summary": "summary",
        }
    )

    builder.add_edge("statistics", "results_merger")
    builder.add_edge("visualization", "results_merger")
    builder.add_edge("summary", "results_merger")
    builder.add_edge("results_merger", END)

    return builder


graph = build_data_analysis_graph().compile()
