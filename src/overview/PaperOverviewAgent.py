import base64
from pathlib import Path
from typing import Any, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import overview.config as cfg

load_dotenv()
Path("./summary").mkdir(exist_ok=True)


class AgentState(TypedDict):
    """Agent state structure."""

    main_query: str
    keywords: list[str]
    search_mess: Sequence[BaseMessage]
    messages: Optional[Sequence[BaseMessage]] = ()


class PaperOverviewAgent:
    def __init__(
        self,
        keywords_model: BaseChatModel,
        search_model: BaseChatModel,
        main_agent: BaseChatModel,
        summary_agent: BaseChatModel,
    ):
        self.keywords_model = keywords_model
        self.main_agent = self.creat_main_agent(main_agent)
        self.search_agent = self.creat_search_agent(search_model)
        self.summary_agent = self.creat_symmary_agent(summary_agent)

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

        if "messages" not in state or state["messages"] is None:
            state["messages"] = []
        state["messages"] = list(state["messages"]) + [system_msg, human_msg, ai_msg]

        save_chat(
            "history/keywords_agent_history.json",
            {"messages": [system_msg, human_msg, ai_msg]},
        )
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

        from overview.tools.memory import save_chat

        system_msg = SystemMessage(content=cfg.SEARCH_AGENT_PROMPT)
        human_msg = HumanMessage(
            content=f"QUERY:{state['main_query']},KEYWORDS:{state['keywords']}"
        )

        if "messages" not in state or state["messages"] is None:
            state["messages"] = []
        search_messages = [system_msg, human_msg] + res["messages"]
        state["messages"] = list(state["messages"]) + search_messages

        save_chat("history/search_agent_history.json", {"messages": search_messages})

        return {
            "search_mess": res["messages"],
            "main_query": state["main_query"],
            "keywords": state["keywords"],
            "messages": state["messages"],
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
        overview_path = Path("./summary") / Path(file_path).with_suffix(".json").name
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

        from langchain_core.messages import SystemMessage

        from overview.tools.memory import save_chat

        system_msg = SystemMessage(content="论文分析阶段已完成")

        if "messages" not in state or state["messages"] is None:
            state["messages"] = []
        state["messages"] = list(state["messages"]) + [system_msg]

        save_chat(
            "history/paper_analysis_agent_history.json", {"messages": [system_msg]}
        )

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
        summ_path = Path("./download")
        file_paths = [p.resolve() for p in summ_path.glob("*.json")]
        data = self._load_json(file_paths)
        res = self.summary_agent.invoke({
            "messages": [
                HumanMessage(content=data),
                HumanMessage(
                    content=f"KEYWORDS:{state['keywords']},MAIN_QUERY:{state['main_query']}"
                ),
            ]
        })
        save_chat("history/summary_agent_history.json", res)
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
            search_mess=[],
            messages=[],
        )
    )


# 第一部，根据主题生成关键词
# 第二步，根据关键词逐一搜索文献，并下载
# 第三步，逐一读取每篇文献，如果和主题密切相关，则保留并将每其关键内容压缩到一个文档（比如一个json格式） with query
# 第四步，把压缩后的文档集合起来写成一篇综述文献
# 综述加参考文献，丢包，搜不到文献，优化搜索
