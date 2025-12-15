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
        "messages": "搜索深度学习的发展,将综述的结果保存到本地文件 @/home/dministrator/git_clone/overview/ov.json"
    })
    from overview.tools.memory import save

    save("history.json", res)
