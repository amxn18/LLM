# RETRIEVERS IN LANGCHAIN 

────────────────────────────────────────────
## 1️ WHAT ARE RETRIEVERS?

Retrievers are components in LangChain responsible for **fetching relevant documents** from a knowledge source based on a query.  
Unlike LLMs that generate text, retrievers **fetch information**, usually stored as embeddings or through APIs.

Retrievers are critical in **Retrieval-Augmented Generation (RAG)** pipelines, where documents are retrieved first and then passed to a language model for answer synthesis.

────────────────────────────────────────────
## 2️ TYPES OF RETRIEVERS

────────────────────────────────────────────
### 1. Wikipedia Retriever
- Uses the public Wikipedia API to fetch text segments related to the query.
- You can configure `top_k_results` (number of articles) and language.
- Ideal for quick factual lookups from a large corpus (Wikipedia) without needing embeddings or a database.

**Workflow**
Wikipedia API → Fetch relevant pages → Return document chunks.

**Pros**
- No need for pre-embedding or vector storage.
- Reliable for general topics and global data.

**Cons**
- Internet connection required.
- Limited to Wikipedia content.

────────────────────────────────────────────
### 2. Vector Store Retriever
- Retrieves documents based on **semantic similarity** using **vector embeddings**.
- The embeddings represent text numerically; retrieval is done by measuring vector distances.
- Uses backends like **Chroma**, **FAISS**, or **Pinecone**.

**Workflow**
Text → Embeddings → Stored in Vector DB → Similarity Search → Return relevant docs.

**Pros**
- High accuracy with semantic matching.
- Can retrieve domain-specific data.
- Works offline once embeddings are stored.

**Cons**
- Requires preprocessing (embedding + storage).
- Needs good embedding models for optimal results.

────────────────────────────────────────────
### 3. Maximal Marginal Relevance (MMR) Retriever
- Variant of vector retrieval that focuses on **diversity + relevance**.
- Balances between fetching similar documents and ensuring they are **not redundant**.

**Working**
It maximizes a combination of:
- Similarity to the query.
- Dissimilarity to previously selected documents.

**Pros**
- Reduces redundancy in results.
- Covers diverse subtopics of a query.

**Cons**
- Slightly more computation compared to simple similarity search.

────────────────────────────────────────────
### 4. Multi-Query Retriever
- Uses an **LLM to rewrite the query** into multiple semantically different subqueries.
- Each subquery searches the vector database separately.
- Combines the results for a broader, more comprehensive retrieval.

**Workflow**
User Query → LLM Generates Variants → Each Variant Searches Vector DB → Results Combined → Return unique documents.

**Pros**
- Significantly improves recall and coverage.
- Handles vague or underspecified queries well.

**Cons**
- Higher latency (multiple searches per query).
- Requires a capable LLM.

────────────────────────────────────────────
### 5. Contextual Compression Retriever
- Compresses documents retrieved by a base retriever to retain only the **most relevant context**.
- Uses a **compressor model** (like an LLM) to refine the output, removing noise or irrelevant details.

**Workflow**
Retriever → Fetch Documents → Compressor (LLM) → Trim content → Return concise documents.

**Pros**
- Reduces token usage by compressing context.
- Improves precision for large, noisy documents.

**Cons**
- Slightly slower due to compression step.
- Requires a good compressor LLM.

────────────────────────────────────────────

# COMPARISON TABLE OF RETRIEVERS

| Retriever Type | Mechanism | Use Case | Pros | Cons |
|----------------|------------|-----------|------|------|
| Wikipedia Retriever | API-based keyword search | General factual retrieval | Simple, quick setup | Limited to Wikipedia |
| Vector Store Retriever | Semantic embedding similarity | Domain-specific retrieval | High semantic accuracy | Requires embeddings |
| MMR Retriever | Balances similarity + diversity | Diverse yet relevant search | Avoids redundancy | Slightly slower |
| Multi-Query Retriever | Multiple LLM-generated subqueries | Complex, broad questions | High recall, robust | Higher latency |
| Contextual Compression Retriever | LLM-based content compression | Noisy or lengthy documents | Cleaner, focused results | More compute required |

────────────────────────────────────────────
## SUMMARY

Retrievers form the information retrieval backbone in LangChain RAG systems.

Each retriever differs by how it interprets the query and ranks results.

Modern pipelines combine retrievers using LCEL and Runnables, replacing old chain mechanisms.

For advanced search systems:
- Use Vector Stores (Chroma, FAISS) for local embeddings.
- Use Vector Databases (Pinecone, Weaviate) for scalable cloud setups.

────────────────────────────────────────────
## FINAL TAKEAWAY

Retrievers bridge the gap between raw knowledge and LLM reasoning.

Choosing the right retriever depends on:
- Dataset size
- Query complexity
- Latency requirements

