from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = CSVLoader(file_path='Langchain\RAG\DOCUMENT LOADERS\Social_Network_Ads.csv')
data = loader.load()
print(data[0])