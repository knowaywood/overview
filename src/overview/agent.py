"""For Agent."""

from deepagents import CompiledSubAgent, create_deep_agent
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


def main_agent(model: str | BaseChatModel):
    """Agent for taking a overview of paper."""
    from overview.tools.search import ddgs_search

    research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

    You have access to an internet search tool as your primary means of gathering information.

    ## `ddgs_search`

    Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
    """

    agent = create_deep_agent(tools=[ddgs_search], system_prompt=research_instructions)
    ...


def search_agent(model: str | BaseChatModel):
    """Agent for search from internet and return the answer."""
    from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
    from deepagents.middleware import FilesystemMiddleware
    from langchain.agents import create_agent
    from langgraph.store.memory import InMemoryStore

    from overview.tools.search import ddgs_search

    store = InMemoryStore()
    agent_graph = create_agent(
        model=model,
        tools=[ddgs_search],
        system_prompt="You are a specialized agent for data analysis...",
        middleware=[
            FilesystemMiddleware(
                backend=lambda rt: CompositeBackend(
                    default=StateBackend(rt),
                    routes={"/memories/": StoreBackend(rt)},
                ),
            ),
        ],
        store=store,
    )
    return agent_graph


def search_subagent(model: str | BaseChatModel):
    """Generate subagent which will be used by main agent."""
    agent_graph = search_agent(model)
    agent_subgraph = CompiledSubAgent(
        name="web-search-agent",
        description="An agent that searches the web for information.",
        runnable=agent_graph,
    )
    return agent_subgraph


if __name__ == "__main__":
    from pprint import pprint

    from langchain_community.chat_models import ChatTongyi

    model = ChatTongyi(model="qwen-max")
    agent = search_agent(model)
    res = agent.invoke({
        "messages": [{"role": "user", "content": "湖南大学是什么大学?"}]
    })
    pprint(res)
