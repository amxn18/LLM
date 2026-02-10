from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch
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
    template = 'Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Summarize the following text \n {text}',
    input_variables= ['text']
)

generateReport = RunnableSequence(prompt1, model, parser)
branchChain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),  # x is output from parser 1 of prompt 1 (Means generate summary only if report size is greater than 500)

    RunnablePassthrough()  # Default Chain
)

chain = RunnableSequence(generateReport, branchChain)
result = chain.invoke({'topic' : 'Russia vs Ukraine'})
print(result)
