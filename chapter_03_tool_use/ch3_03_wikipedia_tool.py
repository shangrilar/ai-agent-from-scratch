import wikipedia

# Listing 3.16
def search_wikipedia(query:str) -> list[str]:
    """Search Wikipedia for a query and return titles of wikipedia pages"""
    search_results = wikipedia.search(query)
    return search_results

def get_wikipedia_page(title:str) -> str:
    """Get a wikipedia page by title"""
    page = wikipedia.page(title, auto_suggest=False)
    return page.content


if __name__ == "__main__":
    # Listing 3.15
    search_results = wikipedia.search("moon")
    print("search_results:")
    print(search_results)

    page = wikipedia.page("Moon", auto_suggest=False)
    print("page content:")
    print(page.content[:100])