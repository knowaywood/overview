"""LLM asert tool."""


def search_raise(content: str):
    """Raise error when all the result of paper not relate to the query.

    Args:
        content (str): question of the interact

    """
    red_start = "\033[31m"
    color_reset = "\033[0m"
    print(f"{red_start}agent raise: {content}{color_reset}")


if __name__ == "__main__":
    search_raise("test")
