import streamlit as st
from frontend.ocr import extractResumeText
from frontend.api_client import call_langserve

st.set_page_config(
    page_title="ATS Resume Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("ATS Resume Analyzer")

job_description = st.text_area("Enter Job Description", height=200)
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

col1, col2 = st.columns(2)

with col1:
    btn1 = st.button("Tell me about the Resume")
    btn2 = st.button("How can I improve my skills for this job?")

with col2:
    btn3 = st.button("What are the missing keywords?")
    btn4 = st.button("Percentage match with Job Description")

if resume_file and job_description.strip():
    with st.spinner("Processing resume..."):
        resume_text = extractResumeText(resume_file)

    if btn1:
        result = call_langserve("/resume/summary", resume_text, job_description)
        st.subheader("Resume Overview")
        st.write(result)

    if btn2:
        result = call_langserve("/resume/improve", resume_text, job_description)
        st.subheader("Skill Improvement Suggestions")
        st.write(result)

    if btn3:
        resultbu = call_langserve("/resume/keywords", resume_text, job_description)
        st.subheader("Missing Keywords")
        st.write(result)

    if btn4:
        result = call_langserve("/resume/match", resume_text, job_description)
        st.subheader("ATS Match Result")
        st.write(result)

elif btn1 or btn2 or btn3 or btn4:
    st.error("Please upload a resume and enter the job description.")
