from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen3-4B-Instruct-2507",
    task = "text-generation", 
    temperature = 0.5
)

model1 = ChatHuggingFace(llm = llm1)  

llm2 = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation",
    temperature = 0.2
)

model2 = ChatHuggingFace(llm = llm2)

prompt1 = PromptTemplate(
    template = "Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "Generate a Linkedin Post about {topic}",
    input_variables= ['topic']
)

parser = StrOutputParser()

parallelChain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model1, parser),
    'linkedin' : RunnableSequence(prompt2, model2, parser)
})

result = parallelChain.invoke({'topic' : "LSTM in Neural Networks"})
print(result)