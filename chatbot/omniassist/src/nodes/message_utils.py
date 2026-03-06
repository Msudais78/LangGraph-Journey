"""Message utility functions."""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.config.models import get_chat_model


def summarize_messages(messages: list, model_name: str = "llama-3.3-70b-versatile") -> str:
    """Summarize a list of messages into a concise summary."""
    if not messages:
        return ""

    model = get_chat_model(model_name, temperature=0)

    text_messages = []
    for m in messages:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        text_messages.append(f"{role}: {m.content}")

    conversation_text = "\n".join(text_messages[-20:])

    summary_prompt = [
        SystemMessage(content="Summarize this conversation concisely, preserving key facts and context:"),
        HumanMessage(content=conversation_text),
    ]

    response = model.invoke(summary_prompt)
    return response.content


def should_summarize(messages: list, threshold: int = 20) -> bool:
    """Check if messages should be summarized based on count."""
    return len(messages) > threshold
