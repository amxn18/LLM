from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import BaseMessage

from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()


def get_model():
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        task="text-generation",
    )
    return ChatHuggingFace(llm=llm)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatNode(state: ChatState):
    model = get_model()
    response = model.invoke(state["messages"])
    return {"messages": [response]}

checkPointer = InMemorySaver()
def createGraph():
    graph = StateGraph(ChatState)
    graph.add_node("chatNode", chatNode)
    graph.add_edge(START, "chatNode")
    graph.add_edge("chatNode", END)

    chatBot = graph.compile(checkpointer=checkPointer)
    return chatBot
chatBot = createGraph()