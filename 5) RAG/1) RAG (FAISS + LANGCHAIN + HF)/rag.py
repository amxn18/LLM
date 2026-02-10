import os 
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.2,
    max_new_tokens=512
)

model = ChatHuggingFace(llm = llm)

embeddings = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")

parser = StrOutputParser()
# Step 1) Data Ingestion
def loadDocument(fileName: str):
    with open("speech.txt", encoding="utf8") as f:
        data = f.read()
    loader = TextLoader(fileName, encoding = "utf8")
    docs = loader.load()
    return docs

def chunkDocs(docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return chunks

documents = loadDocument("speech.txt")
chunks = chunkDocs(documents)

# Step 2) Retrievel 
vectoreStore = FAISS.from_documents(chunks, embeddings)

def retriveSimilarDocs(query: str, k: int = 4):
    topDocs = vectoreStore.similarity_search(query, k = k)
    return topDocs

def generateResponse(query: str):
    Docs = retriveSimilarDocs(query)
    context = "\n\n".join([doc.page_content for doc in Docs])

    prompt = PromptTemplate(
    template = 
    """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Use ten sentences maximum and keep the answer concise.
    <context>
    {context}
    </context>
    Question : {question}
    Answer:
    """,
    input_variables=["context", "question"]
    )

    chain = prompt | model | parser
    response = chain.invoke({
    "context": context,
    "question": query})

    return response

while True:
    query = input("Enter your question related to the document: ")
    
    if not query.strip():
        print("Please enter a valid question.\n")
        continue
    
    print("\nGenerating answer...\n")
    answer = generateResponse(query)
    print("Answer:", answer)
    continue_chat = input("Do you want to ask another question? (yes/no): ").strip().lower()
    
    if continue_chat not in ['yes', 'y']:
        print("\nThank you for using the document Q&A system. Goodbye!")
        break



