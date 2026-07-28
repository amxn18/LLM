from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

import os 
os.environ['LANGSMITH_PROJECT'] = 'Sequential Chain Tracing'


llm1 = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen3-4B-Instruct-2507",
    task = "text-generation",
    temperature = 0.3
)

model1 = ChatHuggingFace(llm=llm1)

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()
chain = prompt1 | model1 | parser | prompt2 | model1 | parser
config = {
    'run_name': 'testing',
    'tags': ['Sequential Chain', 'report-generation', 'summarization'],
    'metatdata': {'model1': 'qwen', 'model2': 'deepseek'}
}
topic = input("Enter your Topic: ")
response = chain.invoke({"topic": topic}, config=config)
print(response)