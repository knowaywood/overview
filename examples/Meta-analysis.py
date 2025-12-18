from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_google_genai import ChatGoogleGenerativeAI

from overview.agent import main_agent, search_subagent
from overview.tools.arxiv import ArxivSearcher
from overview.tools.FS import read_from_local, save2local

load_dotenv()


sub_model = ChatTongyi(model="qwen-max")
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
agent = main_agent(
    model,
    subagents=[search_subagent(sub_model)],
    tools=[read_from_local, ArxivSearcher.search, save2local],
)

if __name__ == "__main__":
    res = agent.invoke({
        "messages": "读取本地文件@/home/dministrator/git_clone/overview/examples/Example/md/1706.03762v7.md,将综述的结果保存到本地文件 @/home/dministrator/git_clone/overview/ov.json"
    })
    from overview.tools.memory import save_chat

    save_chat("history.json", res)
