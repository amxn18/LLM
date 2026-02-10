import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

chatHistory = [
    SystemMessage(content="You are a helpful assistant."),
]

while(True):
    userInput = input("You: ")
    chatHistory.append(HumanMessage(content=userInput)) 
    if userInput.lower() in ['exit', 'quit']:
        print("Exiting the chat. Goodbye!")
        break
    llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="conversational",
    temperature=0.9
    )
    chat = ChatHuggingFace(llm=llm)
    response = chat.invoke(chatHistory)
    chatHistory.append(AIMessage(content=response.content))
    print() 
    print("ChatBot:" ,response.content)   

print("Chat Ended.")
print("Final Chat History:", chatHistory)