"""search tool using DuckDuckGo."""

from langchain_community.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults(output_format="list")


def ddgs_search(query_str: str) -> str:
    """Search the web for the given query string using DuckDuckGo.

    Args:
        query_str (str): The search query string.

    Returns:
        str: The search results.

    """
    return search.invoke(query_str)
