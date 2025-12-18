import base64
from pathlib import Path
from typing import Any, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import overview.config as cfg

load_dotenv()


class AgentState(TypedDict):
    """Agent state structure."""

    main_query: str
    keywords: list[str]
    search_mess: Sequence[BaseMessage]
    main_mess: Sequence[BaseMessage]


class PaperOverviewAgent:
    def __init__(
        self,
        keywords_model: BaseChatModel,
        search_model: BaseChatModel,
        main_agent: BaseChatModel,
    ):
        self.keywords_model = keywords_model
        self.search_model = search_model
        self.main_agent = self.creat_main_agent(main_agent)

        self.search_agent = self.creat_search_agent(self.search_model)
        graph = StateGraph(AgentState)
        graph.add_node("keywords_agent", self.keywords_node)
        graph.add_node("search_agent", self.search_node)

        graph.add_edge(START, "keywords_agent")
        graph.add_edge("keywords_agent", "search_agent")
        graph.add_edge("search_agent", END)
        self.agent = graph.compile()

        # with open("agent_graph.png", "wb") as f:
        #     f.write(self.agent.get_graph().draw_mermaid_png())

    def invoke(self, *args: Any, **kwargs: Any):
        return self.agent.invoke(*args, **kwargs)

    def keywords_node(self, state: AgentState) -> AgentState:
        keywords = self.keywords_model.invoke(
            [SystemMessage(content=cfg.KEYWORD_AGENT_PROMPT)]
            + [HumanMessage(content=state["main_query"])]
        )
        state["keywords"] = [k.strip() for k in str(keywords.content).split(",")]
        print(state)
        return state

    def creat_search_agent(self, model: BaseChatModel):
        from langchain.agents import create_agent

        from overview.tools.agent_raise import search_raise
        from overview.tools.arxiv import ArxivSearcher, download_url

        search_agent = create_agent(
            model=model,
            tools=[ArxivSearcher.search, download_url, search_raise],
        )
        return search_agent

    def search_node(self, state: AgentState):
        res = self.search_agent.invoke({
            "messages": [
                SystemMessage(content=cfg.SEARCH_AGENT_PROMPT),
                HumanMessage(
                    content=f"QUERY:{state['main_query']},KEYWORDS:{state['keywords']}"
                ),
            ]
        })
        return {
            "search_mess": res["messages"],
            "main_query": state["main_query"],
            "keywords": state["keywords"],
        }

    def creat_main_agent(self, model: BaseChatModel):
        from deepagents import create_deep_agent

        from overview.tools.FS import save2local

        main_agent = create_deep_agent(
            model=model, tools=[save2local], system_prompt=cfg.MAIN_AGENT_PROMPT
        )
        return main_agent

    def thread_task(self, model: BaseChatModel, file_path: str):
        with open(file_path, "rb") as pdf_file:
            pdf_data = base64.b64encode(pdf_file.read()).decode("utf-8")
        overview_path = Path(file_path).with_suffix(".json")
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"read the pdf file and analyze its main content。save it overview in {overview_path} use tool save2local.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:application/pdf;base64,{pdf_data}"},
                },
            ]
        )
        res = model.invoke({"messages": [message]})
        chat_path = overview_path.with_name(
            f"{overview_path.stem}_history{overview_path.suffix}"
        )
        save_chat(str(chat_path), res)

    def thread_main_node(self, state: AgentState):
        from concurrent.futures import ThreadPoolExecutor

        dir_path = Path("./download")
        file_paths = [p.resolve() for p in dir_path.glob("*.pdf")]
        num = len(file_paths)

        with ThreadPoolExecutor(max_workers=num) as executor:
            executor.map(self.thread_task, [self.main_agent] * num, file_paths)
        return state

    def summary_agent(self, model: str | BaseChatModel):
        pass


if __name__ == "__main__":
    from langchain_community.chat_models import ChatTongyi
    from langchain_google_genai import ChatGoogleGenerativeAI

    from overview.tools.memory import save_chat

    sub_model = ChatTongyi(model="qwen-max")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    paper_agent = PaperOverviewAgent(
        keywords_model=sub_model, search_model=model, main_agent=model
    )

    his = paper_agent.thread_main_node(
        AgentState(main_query="deeplearning", keywords=[], search_mess=[], main_mess=[])
    )
    print(cfg.paper_dowload)

# 第一部，根据主题生成关键词
# 第二步，根据关键词逐一搜索文献，并下载
# 第三步，逐一读取每篇文献，如果和主题密切相关，则保留并将每其关键内容压缩到一个文档（比如一个json格式）
# 第四步，把压缩后的文档集合起来写成一篇综述文献
