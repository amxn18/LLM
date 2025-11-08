from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen3-4B-Instruct-2507",
    task = "text-generation",
    temperature = 0.8
)

model = ChatHuggingFace(llm = llm)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Generate a joke about {topic}",
    input_variables= ['topic']
)

generateJokeChain = RunnableSequence(prompt1, model, parser)
def countWords(text):
    return len(text.split())

parallelChain = RunnableParallel({
    'Joke' : RunnablePassthrough(),
    'Words' : RunnableLambda(countWords)
})

chain = RunnableSequence(generateJokeChain, parallelChain)
result = chain.invoke({'topic' : 'Sam Altman'})
print(result)
