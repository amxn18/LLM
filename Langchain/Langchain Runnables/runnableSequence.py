from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(   # Runnable 1
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen3-4B-Instruct-2507",
    task = "text-generation", 
    temperature = 0.5
)

model = ChatHuggingFace(llm = llm)  # Runnable 2, 5
parser = StrOutputParser()    # Runnable 3, 6

prompt2 = PromptTemplate(     # Runnable 4 
    template = 'Explain the following joke - {text}',
    input_variables= ['text']
)

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)
result = chain.invoke({'topic' : "AI"})
print(result)