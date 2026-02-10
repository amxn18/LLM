import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import streamlit as st


load_dotenv()

st.header("Basic Summarizer")
user_input = st.text_input("Enter Name Of Research Paper")
if st.button("Generate Summary"):
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="conversational",
        temperature=0.2
    )

    chat = ChatHuggingFace(llm=llm)
    response = chat.invoke(f"Summarize the research paper named {user_input}")
    st.write(response.content)
    st.success("Summary Generated Successfully!")


