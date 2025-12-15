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
    print(f"[+] 搜索查询: {query_str} by DuckDuckGo")
    return search.invoke(query_str)
