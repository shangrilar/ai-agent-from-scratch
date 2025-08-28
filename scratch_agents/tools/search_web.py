import os
from .decorator import tool
from tavily import TavilyClient

@tool
def search_web(query: str, max_results: int = 2) -> str:
    """Search the web for information about a given query"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set. Ensure your .env is loaded before importing this tool or set the env var.")
    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=max_results)
    results = []
    for result in response.get("results"):
        results.append(f"{result.get('title')}\n{result.get('url')}\n{result.get('content')}")
    return "\n\n".join(results)