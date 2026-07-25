import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGSMITH_PROJECT'] = 'RagV1 Tracing'
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen3-4B-Instruct-2507",
    task = "text-generation",
    temperature = 0.3
)

model = ChatHuggingFace(llm = llm)

PDF_PATH = 'test.pdf'

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

embeddingModel = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")
vectorStore = FAISS.from_documents(chunks, embeddingModel)
retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | model | StrOutputParser()

while True:
    user_message = input("Enter your query: ")

    if not user_message.strip():
        print("Please enter a valid question.\n")
        continue
    print("\nGenerating Response...\n")

    response = chain.invoke(user_message.strip())
    print("\nResponse:", response)

    continue_chat = input("Do you want to ask another question? (yes/no): ").strip().lower()

    if continue_chat not in ['yes', 'y']:
        print("\nThank you. Goodbye!")
        break