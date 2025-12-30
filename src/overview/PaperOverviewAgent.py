import base64
from pathlib import Path
from typing import Any, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import overview.config as cfg

load_dotenv()
Path("./summary").mkdir(exist_ok=True)
Path("./history").mkdir(exist_ok=True)


class AgentState(TypedDict):
    """Agent state structure."""

    main_query: str
    keywords: list[str]
    keywords_mess: Sequence[BaseMessage]


class PaperOverviewAgent:
    def __init__(
        self,
        keywords_model: BaseChatModel,
        search_model: BaseChatModel,
        main_agent: BaseChatModel,
        summary_agent: BaseChatModel,
        summ_dir: Path = Path("./summary"),
        history_dir: Path = Path("./history"),
    ):
        self.keywords_model = keywords_model
        self.main_agent = self.creat_main_agent(main_agent)
        self.search_agent = self.creat_search_agent(search_model)
        self.summary_agent = self.creat_symmary_agent(summary_agent)
        self.summ_dir = summ_dir
        self.history_dir = history_dir
        history_dir.mkdir(exist_ok=True)
        summ_dir.mkdir(exist_ok=True)

        graph = StateGraph(AgentState)
        graph.add_node("keywords_agent", self.keywords_node)
        graph.add_node("search_agent", self.search_node)
        graph.add_node("thread_main_agent", self.thread_main_node)
        graph.add_node("summary_agent", self.summary_node)

        graph.add_edge(START, "keywords_agent")
        graph.add_edge("keywords_agent", "search_agent")
        graph.add_edge("search_agent", "thread_main_agent")
        graph.add_edge("thread_main_agent", "summary_agent")
        graph.add_edge("summary_agent", END)
        self.agent = graph.compile()

        with open("agent_graph.png", "wb") as f:
            f.write(self.agent.get_graph().draw_mermaid_png())

    def invoke(self, *args: Any, **kwargs: Any):
        return self.agent.invoke(*args, **kwargs)

    def keywords_node(self, state: AgentState) -> AgentState:
        keywords = self.keywords_model.invoke(
            [SystemMessage(content=cfg.KEYWORD_AGENT_PROMPT)]
            + [HumanMessage(content=state["main_query"])]
        )
        state["keywords"] = [k.strip() for k in str(keywords.content).split(",")]
        print(state)

        from langchain_core.messages import AIMessage

        from overview.tools.memory import save_chat

        system_msg = SystemMessage(content=cfg.KEYWORD_AGENT_PROMPT)
        human_msg = HumanMessage(content=state["main_query"])
        ai_msg = AIMessage(content=keywords.content)

        if "keywords_mess" not in state or state["keywords_mess"] is None:
            state["keywords_mess"] = []
        state["keywords_mess"] = list(state["keywords_mess"]) + [
            system_msg,
            human_msg,
            ai_msg,
        ]

        save_chat(
            f"{self.history_dir}/keywords_agent_history.json",
            {"messages": [system_msg, human_msg, ai_msg]},
        )
        return state

    def creat_search_agent(self, model: BaseChatModel):
        from langchain.agents import create_agent

        from overview.tools.agent_raise import search_raise
        from overview.tools.arxiv import DDGSearcher, download_url

        search_agent = create_agent(
            model=model,
            tools=[DDGSearcher.search, download_url, search_raise],
            system_prompt=cfg.SEARCH_AGENT_PROMPT,
        )
        return search_agent

    def search_node(self, state: AgentState):
        print("search_node")
        res = self.search_agent.invoke({
            "messages": [
                HumanMessage(
                    content=f"QUERY:{state['main_query']},KEYWORDS:{state['keywords']}"
                ),
            ]
        })

        from overview.tools.memory import save_chat

        save_chat(f"{self.history_dir}/search_agent_history.json", {"messages": res})

        return state

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
        overview_path = self.summ_dir / Path(file_path).with_suffix(".json").name
        history_path = self.history_dir / Path(file_path).with_suffix(".json").name
        history_path = history_path.with_name(
            f"{history_path.stem}_history{history_path.suffix}"
        )
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
        save_chat(str(history_path), res)

    def thread_main_node(self, state: AgentState):
        from concurrent.futures import ThreadPoolExecutor

        print("thread_main_node")

        dir_path = Path("./download")
        file_paths = [p.resolve() for p in dir_path.glob("*.pdf")]
        num = len(file_paths)

        with ThreadPoolExecutor(max_workers=num) as executor:
            executor.map(self.thread_task, [self.main_agent] * num, file_paths)
        return state

    def creat_symmary_agent(self, model: BaseChatModel):
        from deepagents import create_deep_agent

        from overview.tools.FS import save2local

        summary_agent = create_deep_agent(
            model=model, tools=[save2local], system_prompt=cfg.SUMMARY_AGENT_PROMPT
        )
        return summary_agent

    def _load_json(self, file_path: list[Path]):

        data = []
        for i in file_path:
            with open(i, "r", encoding="utf-8") as f:
                data.append(f.read())
        return data

    def summary_node(self, state: AgentState) -> AgentState:
        print("summary_node")
        file_paths = [p.resolve() for p in self.summ_dir.glob("*.json")]
        data = self._load_json(file_paths)
        res = self.summary_agent.invoke({
            "messages": [
                HumanMessage(content=data),
                HumanMessage(
                    content=f"KEYWORDS:{state['keywords']},MAIN_QUERY:{state['main_query']}"
                ),
            ]
        })
        save_chat(f"{self.history_dir}/summary_agent_history.json", res)
        return state


if __name__ == "__main__":
    from langchain_community.chat_models import ChatTongyi
    from langchain_google_genai import ChatGoogleGenerativeAI

    from overview.tools.memory import save_chat

    sub_model = ChatTongyi(model="qwen-max")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    flash_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    paper_agent = PaperOverviewAgent(
        keywords_model=sub_model,
        search_model=flash_model,
        main_agent=model,
        summary_agent=model,
    )

    his = paper_agent.invoke(
        AgentState(
            main_query="Lean with machine learning",
            keywords=[],
            keywords_mess=[],
        )
    )


# 第一部，根据主题生成关键词
# 第二步，根据关键词逐一搜索文献，并下载
# 第三步，逐一读取每篇文献，如果和主题密切相关，则保留并将每其关键内容压缩到一个文档（比如一个json格式） with query
# 第四步，把压缩后的文档集合起来写成一篇综述文献
# 综述加参考文献，丢包，搜不到文献，优化搜索
