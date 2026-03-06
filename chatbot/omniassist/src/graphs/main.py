"""Main Orchestrator Graph — the central hub of OmniAssist.

Concepts: StateGraph with Input/Output schema, context_schema, supervisor routing
via Command(goto=...), subgraphs as nodes, memory management.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src.state.main_state import OmniAssistState, OmniAssistInput, OmniAssistOutput
from src.nodes.input_handler import input_handler_node
from src.nodes.supervisor import supervisor_node
from src.nodes.output_handler import output_handler_node
from src.nodes.error_handler import error_handler_node
from src.nodes.memory_manager import memory_manager_node


# ──────────────────────────────────────────────
# SUBGRAPH WRAPPER NODES
# ──────────────────────────────────────────────

def chat_agent_node(state: dict) -> dict:
    """Chat agent — wraps the custom chat agent subgraph."""
    from src.graphs.chat import get_chat_graph
    chat_graph = get_chat_graph()
    result = chat_graph.invoke({
        "messages": state.get("messages", []),
        "user_id": state.get("user_id", "default"),
    })
    return {"messages": result.get("messages", [])[-1:]}


def research_node(state: dict) -> dict:
    """Research subgraph wrapper."""
    from src.graphs.research import graph as research_graph
    result = research_graph.invoke({
        "messages": state.get("messages", []),
        "query": state["messages"][-1].content if state.get("messages") else "",
        "user_id": state.get("user_id", "default"),
    })
    return {
        "messages": result.get("messages", [])[-1:],
        "research_results": result.get("validated_sources", []),
    }


def writing_node(state: dict) -> dict:
    """Writing subgraph wrapper."""
    from src.graphs.writing import graph as writing_graph
    try:
        result = writing_graph.invoke({
            "messages": state.get("messages", []),
            "user_id": state.get("user_id", "default"),
        })
        return {
            "messages": result.get("messages", [])[-1:],
            "draft_content": result.get("draft_content", ""),
        }
    except Exception:
        return {"messages": [AIMessage(content="Writing draft is being prepared. Please check pending approvals.")]}


def task_mgmt_node(state: dict) -> dict:
    """Task management subgraph wrapper."""
    from src.graphs.task_management import graph as task_graph
    try:
        result = task_graph.invoke({
            "messages": state.get("messages", []),
            "user_id": state.get("user_id", "default"),
        })
        return {
            "messages": result.get("messages", [])[-1:],
            "tasks": result.get("tasks", []),
        }
    except Exception:
        return {"messages": [AIMessage(content="Task operation requires your confirmation.")]}


def code_exec_node(state: dict) -> dict:
    """Code execution wrapper — uses Functional API pipeline."""
    from src.workflows.code_pipeline import code_pipeline
    request = state["messages"][-1].content if state.get("messages") else ""
    try:
        result = code_pipeline.invoke(
            request,
            config={"configurable": {"thread_id": state.get("thread_id", "code-default")}},
        )
        output = result.get("output", "")
        code = result.get("code", "")
        response = f"**Code:**\n```python\n{code}\n```\n\n**Output:**\n{output}"
        return {"messages": [AIMessage(content=response)], "code_output": output}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Code execution error: {e}")]}


def data_analysis_node(state: dict) -> dict:
    """Data analysis subgraph wrapper."""
    from src.graphs.data_analysis import graph as data_graph
    result = data_graph.invoke({
        "messages": state.get("messages", []),
        "user_id": state.get("user_id", "default"),
    })
    return {
        "messages": result.get("messages", [])[-1:],
        "data_analysis": {"summary": result.get("summary", "")},
    }


# ──────────────────────────────────────────────
# BUILD MAIN ORCHESTRATOR
# ──────────────────────────────────────────────

def build_main_graph() -> StateGraph:
    """Build the main OmniAssist orchestrator graph."""
    builder = StateGraph(
        state_schema=OmniAssistState,
        input=OmniAssistInput,
        output=OmniAssistOutput,
    )

    builder.add_node("input_handler", input_handler_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("chat_agent", chat_agent_node)
    builder.add_node("research", research_node)
    builder.add_node("writing", writing_node)
    builder.add_node("task_mgmt", task_mgmt_node)
    builder.add_node("code_exec", code_exec_node)
    builder.add_node("data_analysis", data_analysis_node)
    builder.add_node("memory_manager", memory_manager_node)
    builder.add_node("output_handler", output_handler_node)
    builder.add_node("error_handler", error_handler_node)

    # Entry flow
    builder.add_edge(START, "input_handler")
    builder.add_edge("input_handler", "supervisor")

    # supervisor_node returns Command(goto=<agent_name>) — no edges needed
    # All agents flow to memory_manager → output_handler → END
    for agent in ["chat_agent", "research", "writing", "task_mgmt", "code_exec", "data_analysis"]:
        builder.add_edge(agent, "memory_manager")

    builder.add_edge("memory_manager", "output_handler")
    builder.add_edge("output_handler", END)
    builder.add_edge("error_handler", END)

    return builder


def get_main_graph(checkpointer=None, store=None):
    """Return compiled main graph with optional checkpointer and store."""
    builder = build_main_graph()
    return builder.compile(
        checkpointer=checkpointer or InMemorySaver(),
        store=store or InMemoryStore(),
    )


# Default compiled graph for langgraph.json
graph = get_main_graph()
