import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, load_prompt

# Load environment variables (like HF API key)
load_dotenv()

st.header("Basic Research Tool")

# --- User Inputs ---
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# --- Prompt Template ---
template = load_prompt('template.json')

# --- Button to Generate Explanation ---
if st.button("Generate Explanation"):
    st.write("Generating explanation...")

    # Fill prompt
    prompt_text = template.format(
        paper_input=paper_input,
        style_input=style_input,
        length_input=length_input
    )

    # Hugging Face model endpoint
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta", 
        task="text-generation",
        temperature=0.3,
        max_new_tokens=500
    )

    chat = ChatHuggingFace(llm=llm)

    # Generate response
    response = chat.invoke(prompt_text)

    # Display result
    st.subheader("Generated Summary")
    st.write(response.content)
    st.success("Summary Generated Successfully!")
