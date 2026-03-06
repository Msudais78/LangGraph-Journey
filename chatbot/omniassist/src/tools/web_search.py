"""Web search tool using Tavily (with mock fallback)."""

from langchain_core.tools import tool
from src.config.settings import TAVILY_API_KEY


@tool
def web_search(query: str) -> str:
    """Search the web for current information.

    Args:
        query: Search query string
    """
    if not TAVILY_API_KEY or TAVILY_API_KEY.startswith("tvly-REPLACE"):
        # Mock fallback when no Tavily key is configured
        return (
            f"[Mock web search for '{query}']\n"
            f"- Result 1: Overview of {query} from Wikipedia\n"
            f"- Result 2: Latest developments in {query}\n"
            f"- Result 3: Expert analysis of {query}"
        )
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query, max_results=3)
        formatted = []
        for r in results.get("results", []):
            formatted.append(f"- {r['title']}: {r['content'][:200]}")
        return "\n".join(formatted) if formatted else "No results found."
    except Exception as e:
        return f"Search error: {e}"
