"""Supervisor node — routes user messages to the correct specialist agent.

Concepts: LLM-based intent classification, Command(goto=...) for dynamic routing.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command

from src.config.models import get_chat_model
from src.config.prompts import SUPERVISOR_PROMPT


VALID_AGENTS = {"chat_agent", "research", "writing", "task_mgmt", "code_exec", "data_analysis"}


def supervisor_node(state: dict) -> Command:
    """Classify user intent and route to the appropriate agent.

    Uses Command(goto=...) for dynamic routing without predefined edges.
    """
    messages = state.get("messages", [])
    if not messages:
        return Command(
            goto="chat_agent",
            update={"current_intent": "chat", "current_agent": "chat_agent"},
        )

    last_message = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    model = get_chat_model(temperature=0)

    prompt = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=last_message),
    ]

    response = model.invoke(prompt)
    intent = response.content.strip().lower().replace(" ", "_")

    if intent not in VALID_AGENTS:
        intent = "chat_agent"

    return Command(
        goto=intent,
        update={"current_intent": intent, "current_agent": intent},
    )
