import os
from typing_extensions import TypedDict

from dotenv import load_dotenv

# LangChain 相关
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings  # DashScope 向量嵌入 :contentReference[oaicite:13]{index=13}
from langchain_community.vectorstores import FAISS

# LangGraph
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ============ 一、准备全局组件：向量库 + LLM ============

# 1. 加载知识库文本
def load_knowledge_base(path: str = "knowledge_base.txt") -> list[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"知识库文件 {path} 不存在，请先创建")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # 简单起见，按空行分段
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    docs = [Document(page_content=c, metadata={"source": f"chunk_{i}"})
            for i, c in enumerate(chunks)]
    return docs

docs = load_knowledge_base()

# 2. 创建 DashScope 嵌入 + 向量库
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",   # DashScope 文本嵌入模型
    # dashscope_api_key 可从环境变量读取（已在 .env 设置）
)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# 3. 创建通义千问 LLM
llm = ChatTongyi(
    model="qwen-max",
    temperature=0.2,
)

# ============ 二、定义 LangGraph 的 State（状态） ============

class QAState(TypedDict):
    question: str       # 用户问题
    context: str        # 检索到的上下文文本
    answer: str         # 模型回答
    grounded: bool      # 是否「看起来」基于上下文（简单自检结果）

# ============ 三、定义各个节点函数 ============

# 1. 检索节点：根据问题从向量库取出相关片段
def retrieve_node(state: QAState) -> QAState:
    question = state["question"]
    # 新版本 retriever 是一个「可运行对象」，直接用 invoke 调用
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in retrieved_docs)
    return {
        "context": context_text
    }

# 2. 回答节点：强约束「只能用上下文」
system_prompt = """
你是一个非常严谨的知识问答助手。

下面会给你：
- 一段“知识库内容”（context）
- 用户问题

要求：
1. 你的回答只能基于 context 中的内容进行总结或引用。
2. 如果 context 中没有足够信息回答问题，必须回答：“我不知道”，并简单说明原因。
3. 不允许编造 context 中没有提到的事实或数字。
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "知识库内容如下：\n\n{context}\n\n用户问题：{question}\n\n请用中文回答。"),
    ]
)

answer_chain = answer_prompt | llm

def answer_node(state: QAState) -> QAState:
    msg = answer_chain.invoke({
        "context": state["context"],
        "question": state["question"],
    })
    return {"answer": msg.content}

# 3. 自检节点（简单版）：问 LLM “这段回答是不是都基于 context？”
grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个审查助手，负责检查回答是否严格基于给定的知识库内容。"
            "如果回答没有超出知识库，并且没有明显瞎编，请回答 yes；否则回答 no。"
        ),
        (
            "human",
            "知识库内容：\n\n{context}\n\n回答：\n\n{answer}\n\n请只输出 yes 或 no。"
        ),
    ]
)

grade_chain = grade_prompt | llm

def grade_node(state: QAState) -> QAState:
    msg = grade_chain.invoke({
        "context": state["context"],
        "answer": state["answer"],
    })
    verdict = msg.content.strip().lower()
    grounded = verdict.startswith("y")
    return {"grounded": grounded}

# 4. 条件路由函数：根据自检结果决定是否结束
def route_after_grade(state: QAState):
    """
    返回 '__end__' 表示结束，
    返回 'answer' 表示再生成一次（这里为了简单，只结束不循环）。
    """
    if state.get("grounded", False):
        return "__end__"
    # 简单策略：如果判断有问题，就让回答改成“我不知道”
    # 这里直接覆盖 answer，然后结束
    return "__end__"

# ============ 四、构建 StateGraph ============

builder = StateGraph(QAState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("answer", answer_node)
builder.add_node("grade", grade_node)

# 边：START -> retrieve -> answer -> grade -> (根据 route 结束)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", "grade")

# 条件边：grade 完成后，根据 route_after_grade 决定是否结束
builder.add_conditional_edges(
    "grade",
    route_after_grade,
    # 这里我们只需要一个可能的路径 '__end__'，映射到 END
    {"__end__": END},
)

graph = builder.compile()

# ============ 五、CLI 入口 ============

if __name__ == "__main__":
    q = input("请输入你的问题：")
    initial_state: QAState = {
        "question": q,
        "context": "",
        "answer": "",
        "grounded": True,
    }
    final_state = graph.invoke(initial_state)
    print("\n===== 模型回答 =====")
    print(final_state["answer"])
    print("\n（自检结果 grounded =", final_state["grounded"], ")")
