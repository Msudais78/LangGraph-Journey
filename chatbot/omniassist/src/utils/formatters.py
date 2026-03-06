"""Output formatting utilities."""

from __future__ import annotations


def format_research_results(results: list[dict]) -> str:
    if not results:
        return "No research results found."
    lines = []
    for i, result in enumerate(results, 1):
        source = result.get("source", "unknown")
        title = result.get("title", "Untitled")
        content = result.get("content", "")[:200]
        lines.append(f"[{i}] ({source}) {title}\n    {content}")
    return "\n\n".join(lines)


def format_task_list(tasks: list[dict]) -> str:
    if not tasks:
        return "No tasks found."
    lines = []
    for task in tasks:
        status = task.get("status", "pending")
        text = task.get("text", "Untitled task")
        task_id = task.get("id", "?")
        emoji = "✅" if status == "done" else "⏳" if status == "pending" else "🔄"
        lines.append(f"{emoji} [{status}] {text} (ID: {task_id})")
    return "\n".join(lines)


def format_code_output(code: str, output: str, error: str | None = None) -> str:
    parts = [f"**Code:**\n```python\n{code}\n```"]
    if output:
        parts.append(f"**Output:**\n```\n{output}\n```")
    if error:
        parts.append(f"**Error:**\n```\n{error}\n```")
    return "\n\n".join(parts)
