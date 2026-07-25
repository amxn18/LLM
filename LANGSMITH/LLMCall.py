from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen3-4B-Instruct-2507",
    task = "text-generation",
    temperature = 0.3
)

model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
    template = "Give an accurate response to the this particualar question or query from the user" \
    "question: {question}",
    input_variables=['question']
)
parser = StrOutputParser()
chain = prompt | model | parser
ques = input("Enter Your Question: ")
response = chain.invoke({"question": ques})
print(response)