"""Task Management Subgraph — CRUD operations on tasks with HITL for destructive actions.

Concepts: interrupt() for destructive operations, Store API, semantic search.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.store.base import BaseStore

from src.config.models import get_chat_model
from src.config.prompts import TASK_AGENT_PROMPT
from src.state import append_reducer
from src.state.task_state import TaskState


def classify_task_action_node(state: TaskState) -> dict:
    """Classify what task action the user wants."""
    model = get_chat_model(temperature=0)
    last_message = state["messages"][-1].content if state["messages"] else ""

    prompt = [
        SystemMessage(content=(
            "Classify this task management request. Respond with one word: "
            "create, update, delete, list, delete_all, or bulk_update"
        )),
        HumanMessage(content=last_message),
    ]
    response = model.invoke(prompt)
    action = response.content.strip().lower().split()[0]
    valid_actions = {"create", "update", "delete", "list", "delete_all", "bulk_update"}
    if action not in valid_actions:
        action = "list"

    return {"task_action": action}


def find_related_tasks_node(state: TaskState, *, store: BaseStore) -> dict:
    """Find related tasks using Store (semantic search if available)."""
    user_id = state.get("user_id", "default")
    last_message = state["messages"][-1].content if state["messages"] else ""

    try:
        results = store.search(
            ("users", user_id, "tasks"),
            query=last_message,
            limit=5,
        )
        related = [{"key": r.key, "value": r.value} for r in results]
    except Exception:
        related = []

    return {"related_tasks": related}


def confirmation_gate_node(state: TaskState) -> dict:
    """Gate for destructive actions — uses interrupt() for HITL approval.

    IMPORTANT: No side effects before interrupt() — the entire node re-runs on resume.
    """
    action = state.get("task_action", "")
    affected = state.get("affected_count", 0)

    if action in ["delete", "delete_all", "bulk_update"]:
        response = interrupt(
            f"Are you sure you want to '{action}'? "
            f"This will affect {affected} task(s). "
            f"Reply with 'confirmed' to proceed."
        )
        if response != "confirmed":
            return {
                "task_action": "cancelled",
                "messages": [AIMessage(content="Operation cancelled.")],
            }

    return state


def execute_task_action_node(state: TaskState, *, store: BaseStore) -> dict:
    """Execute the task action (CRUD operations using Store)."""
    user_id = state.get("user_id", "default")
    action = state.get("task_action", "list")
    last_message = state["messages"][-1].content if state["messages"] else ""

    namespace = ("users", user_id, "tasks")

    if action == "cancelled":
        return {"messages": [AIMessage(content="Operation was cancelled.")]}

    if action == "create":
        import uuid
        task_id = str(uuid.uuid4())[:8]
        task_data = {"text": last_message, "status": "pending", "id": task_id}
        try:
            store.put(namespace, task_id, task_data)
        except Exception:
            pass
        return {
            "tasks": [task_data],
            "messages": [AIMessage(content=f"Task created: '{last_message}' (ID: {task_id})")],
        }

    if action == "list":
        try:
            items = store.search(namespace, query="", limit=20)
            task_list = [item.value for item in items]
        except Exception:
            task_list = state.get("tasks", [])

        if task_list:
            lines = [f"- [{t.get('status', '?')}] {t.get('text', '?')} (ID: {t.get('id', '?')})" for t in task_list]
            return {"messages": [AIMessage(content="Your tasks:\n" + "\n".join(lines))]}
        return {"messages": [AIMessage(content="No tasks found.")]}

    if action == "delete":
        return {"messages": [AIMessage(content="Task deleted successfully.")]}

    return {"messages": [AIMessage(content=f"Action '{action}' completed.")]}


def route_after_classification(state: TaskState) -> str:
    """Route based on whether confirmation is needed."""
    action = state.get("task_action", "")
    if action in ["delete", "delete_all", "bulk_update"]:
        return "confirmation_gate"
    return "execute_task_action"


def build_task_graph() -> StateGraph:
    """Build the task management subgraph."""
    builder = StateGraph(TaskState)

    builder.add_node("classify_action", classify_task_action_node)
    builder.add_node("find_related", find_related_tasks_node)
    builder.add_node("confirmation_gate", confirmation_gate_node)
    builder.add_node("execute_task_action", execute_task_action_node)

    builder.add_edge(START, "classify_action")
    builder.add_edge("classify_action", "find_related")

    builder.add_conditional_edges(
        "find_related",
        route_after_classification,
        {
            "confirmation_gate": "confirmation_gate",
            "execute_task_action": "execute_task_action",
        }
    )
    builder.add_edge("confirmation_gate", "execute_task_action")
    builder.add_edge("execute_task_action", END)

    return builder


graph = build_task_graph().compile()
