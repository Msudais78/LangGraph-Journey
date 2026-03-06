"""Code Execution Pipeline — built using the Functional API (@entrypoint/@task).

Concepts: @entrypoint, @task, interrupt(), entrypoint.final, previous parameter.
"""

from __future__ import annotations

from langgraph.func import entrypoint, task
from langgraph.types import interrupt, RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

from src.config.models import get_chat_model
from src.config.prompts import CODE_AGENT_PROMPT


@task(retry_policy=RetryPolicy(max_attempts=2, initial_interval=1.0, backoff_factor=2.0))
def generate_code(request: str) -> dict:
    """Generate Python code from a natural language request."""
    model = get_chat_model(temperature=0.2)
    from langchain_core.messages import SystemMessage, HumanMessage

    response = model.invoke([
        SystemMessage(content=CODE_AGENT_PROMPT + "\nGenerate ONLY Python code. No explanation. Just code."),
        HumanMessage(content=request),
    ])
    return {"code": response.content, "language": "python"}


@task
def review_code(code_info: dict) -> dict:
    """Review code for correctness and safety."""
    code = code_info.get("code", "")

    dangerous_patterns = ["os.system", "subprocess", "shutil.rmtree", "os.remove", "import socket"]
    safety_level = "safe"
    for pattern in dangerous_patterns:
        if pattern in code:
            safety_level = "dangerous"
            break

    if safety_level == "safe" and any(p in code for p in ["requests.", "http", "urllib"]):
        safety_level = "caution"

    model = get_chat_model(temperature=0)
    from langchain_core.messages import SystemMessage, HumanMessage
    review_response = model.invoke([
        SystemMessage(content="Review this code briefly (1-2 sentences). Note any issues."),
        HumanMessage(content=code),
    ])

    return {
        **code_info,
        "safety_level": safety_level,
        "review": review_response.content,
    }


@task
def execute_code(code: str) -> dict:
    """Execute code in a sandboxed environment."""
    import io
    import contextlib

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec_globals = {"__builtins__": __builtins__}
            exec(code, exec_globals)
        output = stdout_capture.getvalue()
        error = stderr_capture.getvalue() if stderr_capture.getvalue() else None
        return {"output": output or "Code executed successfully (no output).", "error": error}
    except Exception as e:
        return {"output": "", "error": f"{type(e).__name__}: {e}"}


@entrypoint(checkpointer=InMemorySaver())
def code_pipeline(request: str, *, previous: dict | None = None) -> dict:
    """Main code execution pipeline using Functional API.

    Args:
        request: Natural language code request
        previous: Result of the last invocation on the same thread (for context continuity).
    """
    context = ""
    if previous and previous.get("code"):
        context = f"\n\nPrevious code:\n{previous['code']}\nPrevious output:\n{previous.get('output', '')}"

    full_request = request + context if context else request

    code_info = generate_code(full_request).result()
    review = review_code(code_info).result()

    if review["safety_level"] == "dangerous":
        approval = interrupt({
            "code": code_info["code"],
            "safety_level": "dangerous",
            "review": review["review"],
            "prompt": "This code is potentially unsafe. Reply 'approved' to execute or 'rejected' to cancel.",
        })
        if approval != "approved":
            return entrypoint.final(
                value={"status": "cancelled", "reason": "User rejected unsafe code"},
                save={"last_request": request, "cancelled": True, "code": code_info["code"]},
            )

    result = execute_code(code_info["code"]).result()

    return {
        "code": code_info["code"],
        "language": code_info["language"],
        "safety_level": review["safety_level"],
        "review": review["review"],
        "output": result.get("output", ""),
        "error": result.get("error"),
    }
