from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from dotenv import load_dotenv
import os

load_dotenv()

# chatTemplate = ChatPromptTemplate([
#     SystemMessage(content="You are a helpful {doamin} expert."),
#     HumanMessage(content = 'Explain in simple terms about {topic}.')
#     Problem : # Generated Prompt Messages: messages=[SystemMessage(content='You are a helpful {doamin} expert.', 
#     additional_kwargs={}, response_metadata={}), 
#     HumanMessage(content='Explain in simple terms about {topic}.', 
#     additional_kwargs={}, response_metadata={})]
# ])

chatTemplate = ChatPromptTemplate([
    ('system', "You are a helpful {doamin} expert."),
    ('human', 'Explain in simple terms about {topic}.')

    # Output: Generated Prompt Messages: 
    # messages=[SystemMessage(content='You are a helpful {doamin} expert.', 
    # additional_kwargs={}, response_metadata={}), 
    # HumanMessage(content='Explain in simple terms about {topic}.', 
    # additional_kwargs={}, response_metadata={})]
])
prompt = chatTemplate.invoke({
    "doamin": "science",
    "topic": "quantum computing"})

print("Generated Prompt Messages:", prompt)