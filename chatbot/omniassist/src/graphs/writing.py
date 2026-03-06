"""Writing Subgraph — handles content creation with iterative review cycles.

Concepts: Cycles, interrupt() for HITL, Command(resume=...).
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy

from src.config.models import get_chat_model
from src.config.prompts import WRITING_AGENT_PROMPT
from src.state.writing_state import WritingState


def analyze_request_node(state: WritingState) -> dict:
    """Analyze the writing request to determine content type and approach."""
    model = get_chat_model(temperature=0)
    last_message = state["messages"][-1].content if state["messages"] else ""

    prompt = [
        SystemMessage(content=(
            "Classify this writing request. Respond with one word: "
            "email, article, summary, report, or other"
        )),
        HumanMessage(content=last_message),
    ]
    response = model.invoke(prompt)
    content_type = response.content.strip().lower().split()[0]
    valid_types = {"email", "article", "summary", "report", "other"}
    if content_type not in valid_types:
        content_type = "other"

    return {
        "writing_request": last_message,
        "content_type": content_type,
        "iteration_count": 0,
    }


def generate_draft_node(state: WritingState) -> dict:
    """Generate or revise the content draft."""
    model = get_chat_model(temperature=0.7)

    iteration = state.get("iteration_count", 0)
    feedback = state.get("human_feedback")

    if iteration == 0 or not state.get("draft_content"):
        prompt = [
            SystemMessage(content=WRITING_AGENT_PROMPT),
            HumanMessage(content=(
                f"Create a {state.get('content_type', 'text')} based on this request:\n"
                f"{state.get('writing_request', '')}"
            )),
        ]
    else:
        prompt = [
            SystemMessage(content=WRITING_AGENT_PROMPT),
            HumanMessage(content=(
                f"Revise this {state.get('content_type', 'text')} based on feedback.\n\n"
                f"Original request: {state.get('writing_request', '')}\n\n"
                f"Current draft:\n{state.get('draft_content', '')}\n\n"
                f"Feedback: {feedback}\n\n"
                "Please produce an improved version."
            )),
        ]

    response = model.invoke(prompt)

    return {
        "draft_content": response.content,
        "revision_history": [response.content],
        "iteration_count": iteration + 1,
    }


def quality_review_node(state: WritingState) -> dict:
    """Auto-review the draft for quality."""
    model = get_chat_model(temperature=0)
    draft = state.get("draft_content", "")

    prompt = [
        SystemMessage(content="Rate this writing 1-10 for quality. Respond with just the number."),
        HumanMessage(content=draft[:2000]),
    ]

    try:
        response = model.invoke(prompt)
        score = float(response.content.strip().split()[0])
    except Exception:
        score = 7.0

    return {"quality_score": score}


def human_review_node(state: WritingState) -> dict:
    """Human-in-the-loop review using modern interrupt() function.

    IMPORTANT: When execution resumes via Command(resume=...),
    the ENTIRE node re-executes from the beginning. interrupt()
    then returns the resume value. No side effects before interrupt()!
    """
    draft = state.get("draft_content", "")
    quality_score = state.get("quality_score", 0)

    feedback = interrupt({
        "draft": draft,
        "quality_score": quality_score,
        "iteration": state.get("iteration_count", 0),
        "prompt": "Review this draft. Respond with: {'action': 'approve'|'revise'|'restart', 'feedback': '...'}",
    })

    action = feedback.get("action", "approve") if isinstance(feedback, dict) else "approve"
    fb_text = feedback.get("feedback", "") if isinstance(feedback, dict) else str(feedback)

    return {
        "action": action,
        "human_feedback": fb_text,
    }


def route_after_review(state: WritingState) -> str:
    """Route based on human review action."""
    action = state.get("action", "approve")
    iteration = state.get("iteration_count", 0)

    if action == "approve":
        return "finalize"
    elif action == "restart":
        return "generate_draft"
    elif action == "revise" and iteration < 5:
        return "generate_draft"
    else:
        return "finalize"


def finalize_node(state: WritingState) -> dict:
    """Finalize the writing and produce the output message."""
    draft = state.get("draft_content", "")
    return {
        "messages": [AIMessage(content=f"Here's your finalized content:\n\n{draft}")],
        "draft_content": draft,
    }


def build_writing_graph() -> StateGraph:
    """Build the writing subgraph with review cycles and HITL."""
    builder = StateGraph(WritingState)

    builder.add_node("analyze_request", analyze_request_node)
    builder.add_node("generate_draft", generate_draft_node, retry=RetryPolicy(max_attempts=2))
    builder.add_node("quality_review", quality_review_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "analyze_request")
    builder.add_edge("analyze_request", "generate_draft")
    builder.add_edge("generate_draft", "quality_review")
    builder.add_edge("quality_review", "human_review")

    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "generate_draft": "generate_draft",
            "finalize": "finalize",
        }
    )
    builder.add_edge("finalize", END)

    return builder


graph = build_writing_graph().compile()
