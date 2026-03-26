from tavily import TavilyClient
import os


def search_web(
    query: str,
    max_results: int = 5,
    topic: str = "general",
    time_range: str | None = None,
) -> list | str:
    """Search the web using Tavily API.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        topic: Search topic - 'general' or 'news'
        time_range: Time range filter (e.g., 'day', 'week', 'month', 'year')
    """
    try:
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        kwargs = {
            "query": query,
            "max_results": max_results,
            "topic": topic,
        }
        if time_range:
            kwargs["time_range"] = time_range

        response = client.search(**kwargs)
        return response.get("results", [])
    except Exception as e:
        return f"Search error: {str(e)}"
