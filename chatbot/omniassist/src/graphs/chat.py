"""Chat Agent — demonstrates both prebuilt and custom StateGraph approaches.

Uses Groq LLMs (llama-3.3-70b-versatile) instead of OpenAI.
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import warnings
from langgraph.types import Command

from src.config.prompts import CHAT_AGENT_PROMPT
from src.config.models import get_chat_model
from src.tools import CHAT_TOOLS, CODE_TOOLS


# ──────────────────────────────────────────────────
# APPROACH 1: Quick bootstrap with create_react_agent
# ──────────────────────────────────────────────────
def build_simple_chat_agent():
    """Build a simple chat agent using the prebuilt create_react_agent helper."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(
            model=get_chat_model(),
            tools=CHAT_TOOLS,
            prompt=CHAT_AGENT_PROMPT,
        )


# ──────────────────────────────────────────────────
# APPROACH 2: Full custom StateGraph for deep control
# ──────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    current_agent: str


def chat_model_node(state: ChatState) -> dict:
    """Call the LLM with tools bound."""
    model = get_chat_model()

    # Keep last 20 messages to avoid context overflow
    messages = state["messages"][-20:] if len(state.get("messages", [])) > 20 else state.get("messages", [])

    # Prepend system message
    messages_with_system = [SystemMessage(content=CHAT_AGENT_PROMPT)] + messages

    # Dynamic tool calling — add code tools if code execution is contextually relevant
    tools = list(CHAT_TOOLS)
    last_msg = state["messages"][-1].content if state["messages"] else ""
    if any(kw in last_msg.lower() for kw in ["code", "python", "script", "program", "execute"]):
        tools.extend(CODE_TOOLS)

    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages_with_system)

    return {"messages": [response], "current_agent": "chat_agent"}


def should_use_tools(state: ChatState) -> str:
    """Conditional edge: check if the last message has tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_custom_chat_agent() -> StateGraph:
    """Build a custom chat agent StateGraph with full control."""
    all_possible_tools = list(CHAT_TOOLS) + list(CODE_TOOLS)
    tool_node = ToolNode(all_possible_tools)

    builder = StateGraph(ChatState)

    builder.add_node("chat_model", chat_model_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "chat_model")
    builder.add_conditional_edges("chat_model", should_use_tools, {"tools": "tools", END: END})
    builder.add_edge("tools", "chat_model")

    return builder


def get_chat_graph(checkpointer=None):
    """Return compiled chat agent graph."""
    builder = build_custom_chat_agent()
    return builder.compile(checkpointer=checkpointer)
