"""Knowledge base / vector store search tool (mock)."""

from langchain_core.tools import tool


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for relevant information.

    Args:
        query: Search query
    """
    return f"Knowledge base results for '{query}': Relevant information about the topic has been found in the internal database."
