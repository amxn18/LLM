from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm)
parser = StrOutputParser()

prompt = PromptTemplate(
    template = "Generate A Summary of following poem - \n {text}",
    input_variables=['text']
)

loader = TextLoader('Langchain\RAG\DOCUMENT LOADERS\cricket.txt', encoding='utf-8')
docs = loader.load()

loaderChain = RunnableLambda(lambda _: loader.load()[0].page_content)
chain = loaderChain| prompt | model | parser
result = chain.invoke(None)

print(result)