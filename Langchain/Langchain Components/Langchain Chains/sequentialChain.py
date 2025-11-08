from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation",
    temperature= 0.2,
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = "Generate a detailed report on the following topic: {topic}",
    input_variables = ["topic"],
)

prompt2 = PromptTemplate(
    template = "Summarize the following report into key bullet points:\n\n{report}",
    input_variables = ["report"],
)

prompt3 = PromptTemplate(
    template = "Create a catchy title for the following bullet points:\n\n{bullet_points}",
    input_variables = ["bullet_points"]
)
parser = StrOutputParser()

chain = prompt1 | model | parser| prompt2 | model | parser| prompt3 | model | parser

result = chain.invoke({"topic": "The impact of artificial intelligence on modern education"})
print(result)