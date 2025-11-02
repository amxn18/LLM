from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello, who won the world series in 2020?"),
]

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="conversational",
    temperature=0.4
)
chat = ChatHuggingFace(llm=llm)
response = chat.invoke(messages)

messages.append(AIMessage(content=response.content))
print(messages) 