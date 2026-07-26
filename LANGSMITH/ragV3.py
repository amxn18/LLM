import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

from langsmith import traceable

load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "RagV3 Tracing (Latency Fix By Indexing)"

PDF_PATH = "test.pdf"

INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# LLM
# --------------------------------------------------

@traceable(name="getModel")
def getModel():
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        task="text-generation",
        temperature=0.3,
    )
    return ChatHuggingFace(llm=llm)


model = getModel()


# --------------------------------------------------
# PDF Loading
# --------------------------------------------------

@traceable(
    name="loadPDF",
    tags=["pdf", "loader"],
    metadata={"loader": "PyPDFLoader"},
)
def loadPDF(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


# --------------------------------------------------
# Chunking
# --------------------------------------------------

@traceable(name="docsChunker")
def docsChunker(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

def getEmbeddingModel(model_name):
    return HuggingFaceEndpointEmbeddings(
        repo_id=model_name
    )


# --------------------------------------------------
# Vector Store
# --------------------------------------------------

@traceable(name="createVectorStore")
def createVectorStore(chunks, embed_model_name):
    embedding_model = getEmbeddingModel(embed_model_name)
    return FAISS.from_documents(chunks, embedding_model)


# --------------------------------------------------
# Index Utilities
# --------------------------------------------------

def _file_fingerprint(path: str):
    p = Path(path)

    h = hashlib.sha256()

    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return {
        "sha256": h.hexdigest(),
        "size": p.stat().st_size,
        "mtime": int(p.stat().st_mtime),
    }


def _index_key(
    pdf_path,
    chunk_size,
    chunk_overlap,
    embed_model_name,
):
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }

    return hashlib.sha256(
        json.dumps(meta, sort_keys=True).encode()
    ).hexdigest()


# --------------------------------------------------
# Load Existing Index
# --------------------------------------------------

@traceable(name="load_index", tags=["index"])
def load_index_run(index_dir, embed_model_name):
    embedding_model = getEmbeddingModel(embed_model_name)

    return FAISS.load_local(
        str(index_dir),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


# --------------------------------------------------
# Build New Index
# --------------------------------------------------

@traceable(name="build_index", tags=["index"])
def build_index_run(
    pdf_path,
    index_dir,
    chunk_size,
    chunk_overlap,
    embed_model_name,
):
    docs = loadPDF(pdf_path)

    chunks = docsChunker(
        docs,
        chunk_size,
        chunk_overlap,
    )

    vectorstore = createVectorStore(
        chunks,
        embed_model_name,
    )

    index_dir.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(index_dir))

    (index_dir / "meta.json").write_text(
        json.dumps(
            {
                "pdf_path": os.path.abspath(pdf_path),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": embed_model_name,
            },
            indent=2,
        )
    )

    return vectorstore


# --------------------------------------------------
# Dispatcher
# --------------------------------------------------

def load_or_build_index(
    pdf_path,
    chunk_size=1000,
    chunk_overlap=150,
    embed_model_name=EMBED_MODEL_NAME,
    force_rebuild=False,
):
    key = _index_key(
        pdf_path,
        chunk_size,
        chunk_overlap,
        embed_model_name,
    )

    index_dir = INDEX_ROOT / key

    if index_dir.exists() and not force_rebuild:
        print("Loading cached FAISS index...")
        return load_index_run(
            index_dir,
            embed_model_name,
        )

    print("Building new FAISS index...")

    return build_index_run(
        pdf_path,
        index_dir,
        chunk_size,
        chunk_overlap,
        embed_model_name,
    )


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer ONLY from the provided context. "
            "If not found, say you don't know.",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}",
        ),
    ]
)


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# --------------------------------------------------
# Setup Pipeline
# --------------------------------------------------

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(
    pdf_path,
    chunk_size=1000,
    chunk_overlap=150,
    embed_model_name=EMBED_MODEL_NAME,
    force_rebuild=False,
):
    return load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=force_rebuild,
    )


# --------------------------------------------------
# Query
# --------------------------------------------------

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(
    pdf_path,
    question,
    chunk_size=1000,
    chunk_overlap=150,
    embed_model_name=EMBED_MODEL_NAME,
    force_rebuild=False,
):
    vectorstore = setup_pipeline(
        pdf_path,
        chunk_size,
        chunk_overlap,
        embed_model_name,
        force_rebuild,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    parallel = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    chain = (
        parallel
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke(
        question,
        config={
            "run_name": "pdf_rag_query",
            "tags": ["qa"],
            "metadata": {"k": 4},
        },
    )


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    print("PDF RAG Ready!")
    while True:
        user_message = input("Enter your query: ")
        if not user_message.strip():
            print("Please enter a valid question.\n")
            continue
        print("\nGenerating Response...\n")

        response  = setup_pipeline_and_query(PDF_PATH, user_message)
        print("\nResponse:", response)

        continue_chat = input("Do you want to ask another question? (yes/no): ").strip().lower()

        if continue_chat not in ['yes', 'y']:
            print("\nThank you. Goodbye!")
            break