import wikipedia
from .decorator import tool

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a given query"""
    return wikipedia.search(query)

@tool
def get_wikipedia_page(page_name: str) -> str:
    """Get the content of a Wikipedia page"""
    return wikipedia.page(page_name).content