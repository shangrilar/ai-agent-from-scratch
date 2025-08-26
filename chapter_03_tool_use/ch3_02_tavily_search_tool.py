# Listing 3.13
import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    
def search_web(query: str) -> str:
    """Search the web for the given query."""
    response = tavily_client.search(query, max_results=2, chunks_per_source=2)
    return response.get("results")

# Listing 3.14
print(search_web("Kipchoge's marathon world record"))