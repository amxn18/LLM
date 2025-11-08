from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(   
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen3-4B-Instruct-2507",
    task = "text-generation", 
    temperature = 0.5
)

model = ChatHuggingFace(llm = llm) 
parser = StrOutputParser()   

prompt2 = PromptTemplate(     
    template = 'Explain the following joke - {text}',
    input_variables= ['text']
)

generateJokeChain = RunnableSequence(prompt1, model, parser)
parallelChain = RunnableParallel({
    'Joke': RunnablePassthrough(),
    'Explanation' : RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(generateJokeChain, parallelChain)
result = chain.invoke({"topic" : "Elon Musk"})
print(result)