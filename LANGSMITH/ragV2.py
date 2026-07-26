import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langsmith import traceable

load_dotenv()

os.environ['LANGSMITH_PROJECT'] = 'RagV2 Tracing'
PDF_PATH = 'test.pdf'

@traceable(name = "getModel")
def getModel():
    llm = HuggingFaceEndpoint(
        repo_id= "Qwen/Qwen3-4B-Instruct-2507",
        task = "text-generation",
        temperature = 0.3)
    model = ChatHuggingFace(llm = llm)
    return model

@traceable(name = "loadPDF", tags = ['pdf', 'loader'], metadata={'loader': 'pyPDFLoader'})
def loadPDF(path: str):
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    return docs

@traceable(name = "docsChunker")
def docsChunker(docs, chunk_size = 1000, chunk_overlap = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks

@traceable(name = "createVectorStore")
def createVectorStore(chunks):
    embeddingModel = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")
    vectorStore = FAISS.from_documents(chunks, embeddingModel)
    return vectorStore

@traceable(name = "getRetriever")
def getRetriever(vectorStore):
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever

@traceable(name = "setupPipeline")
def setupPipeline(pdf_path: str):
    docs = loadPDF(pdf_path)
    chunks = docsChunker(docs)
    vectorStore = createVectorStore(chunks)
    retriever = getRetriever(vectorStore)
    return retriever

retriever = setupPipeline(PDF_PATH)
model = getModel()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})
config = {"run_name": "pdf_rag_query_V2"}
chain = parallel | prompt | model | StrOutputParser()

while True:
    user_message = input("Enter your query: ")

    if not user_message.strip():
        print("Please enter a valid question.\n")
        continue
    print("\nGenerating Response...\n")

    response = chain.invoke(user_message.strip(), config=config)
    print("\nResponse:", response)

    continue_chat = input("Do you want to ask another question? (yes/no): ").strip().lower()

    if continue_chat not in ['yes', 'y']:
        print("\nThank you. Goodbye!")
        break