"""Email tool (mock implementation requiring HITL approval)."""

from langchain_core.tools import tool


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. This action requires human approval.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body text
    """
    return f"Email sent to {to} with subject '{subject}'."
