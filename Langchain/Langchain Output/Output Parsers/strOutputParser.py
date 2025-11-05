from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv() 

llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task = "text-generation",
    temperature = 0.3
)

model = ChatHuggingFace(llm=llm)

# 1st Prompt --> Detailed Report
template1 = PromptTemplate(
    template = "Write a detailed about the {topic}.",
    input_variables = ["topic"]
)

# 2nd Prompt --> Summary
template2 = PromptTemplate(
    template = "Write a 5 line summary on the following text. /n {text}.",
    input_variables = ["topic"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2| model | parser 
result = chain.invoke({'topic' : 'Overpopulation in India'})

print("Final 5 Line Summary:\n", result)

 