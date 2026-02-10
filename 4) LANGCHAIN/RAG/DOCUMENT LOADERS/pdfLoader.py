from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader 
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


loader = PyPDFLoader("Langchain\RAG\DOCUMENT LOADERS\pdf2.pdf")
docs = loader.load()
print(docs)
print(len(docs))
print(docs[0].page_content)