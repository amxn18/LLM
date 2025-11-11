# VECTOR STORES AND VECTOR DATABASES 
────────────────────────────────────────────────────────
# 1. WHAT ARE VECTORS AND EMBEDDINGS?
────────────────────────────────────────────────────────

In RAG systems, raw text cannot be compared directly.  
Instead, text is converted into numerical vectors using an embedding model.

Key properties:
- Semantically similar texts → vectors close together
- Unrelated texts → vectors far apart
- Embeddings preserve meaning, not keywords

In this project, the embedding model used is:
- A sentence-transformer model that produces dense numerical embeddings for each document.

Embeddings enable fast, meaningful similarity search inside a vector database.

────────────────────────────────────────────────────────
# 2. WHAT IS A VECTOR DATABASE?
────────────────────────────────────────────────────────

A vector database is a real storage and indexing system designed for:
- Storing embeddings
- Running fast similarity search (ANN search)
- Handling persistence
- Performing metadata filtering
- Managing large-scale vector collections

Examples include:
- ChromaDB
- Pinecone
- Weaviate
- Milvus
- Qdrant
- Elasticsearch ANN

A vector database is the actual backend engine performing storage, indexing, and retrieval.

────────────────────────────────────────────────────────
# 3. WHAT IS A VECTOR STORE IN LANGCHAIN?
────────────────────────────────────────────────────────

A **vector store** is NOT a database.  
It is a LangChain wrapper that provides a unified interface to interact with ANY vector database.

Purpose:
- Add documents  
- Add embeddings  
- Search for similar documents  
- Delete documents  
- Update documents  
- Convert into a retriever  

LangChain provides vector stores such as:
- Chroma
- FAISS
- Pinecone wrapper
- Weaviate wrapper

The vector store sits between your Python code and the underlying vector database.

────────────────────────────────────────────────────────
# 4. DIFFERENCE: VECTOR STORE VS VECTOR DATABASE
────────────────────────────────────────────────────────

| Concept           | Vector Store (LangChain)                       | Vector Database (Actual System)                       |
|-------------------|--------------------------------------------------|--------------------------------------------------------|
| What it is        | Wrapper/abstraction                              | Real backend storage and search engine                |
| Purpose           | Provide consistent API                           | Perform ANN search, store vectors, scale to large data |
| Lives where?      | In your Python environment                        | Locally/on disk or in the cloud                       |
| Handles indexing? | No                                               | Yes                                                   |
| Handles scaling?  | No                                               | Yes                                                   |
| Provides search?  | Uses database functions                          | Native high-performance search                        |
| Examples          | LangChain Chroma wrapper                         | Chroma, Pinecone, Milvus, Qdrant                      |

In summary:
- A vector database is the actual technology.
- A vector store is the tool you use **inside LangChain** to interact with that technology.

────────────────────────────────────────────────────────
# 5. WORKFLOW OVERVIEW
────────────────────────────────────────────────────────

Your code follows the standard RAG embedding → storage → retrieval workflow:

1. **Create embeddings**  
   Convert text into vector format using an embedding model.

2. **Create documents**  
   Each document has:
   - Content
   - Metadata

3. **Initialize the vector store**  
   Chroma is used as the underlying vector database.

4. **Add documents**  
   Text + metadata → embedded → stored in the database.

5. **Similarity search**  
   Query text → embedded → compared → closest vectors returned.

6. **Metadata filtering**  
   Allows filtering results by custom metadata tags.

7. **Updating documents**  
   Replaces content and embedding at the same document ID.

8. **Deleting documents**  
   Removes entries by ID.

9. **Persistence**  
   Stored to disk inside a folder so the vector database can reload later.

────────────────────────────────────────────────────────
# 6. CHROMA DATABASE EXPLAINED
────────────────────────────────────────────────────────

Chroma is an open-source vector database with features designed for local and small-to-medium RAG systems.

Key characteristics:
- Fast in-memory performance  
- Supports persistence (stores data on disk)  
- Easy to integrate with LangChain  
- Lightweight, no external servers required  
- Good for development and medium-sized applications  

Chroma stores:
- Embeddings
- Documents
- Metadata
- Document IDs

Your code uses:
- A named collection
- A persistent directory

This means data stays saved across sessions.

────────────────────────────────────────────────────────
# 7. DOCUMENTS AND METADATA
────────────────────────────────────────────────────────

Each document contains:
- Page content (raw text)
- Metadata (helpful for filtering and retrieval)

Metadata in your dataset includes IPL team names such as:
- Royal Challengers Bangalore
- Mumbai Indians
- Chennai Super Kings

Metadata is extremely useful in:
- Retrieval filtering
- Categorization
- Multi-tenant RAG systems
- Domain-based query routing

────────────────────────────────────────────────────────
# 8. SIMILARITY SEARCH
────────────────────────────────────────────────────────

Similarity search works by comparing the vector of the query against stored vectors.

Two main operations are used:

1. Simple similarity search  
   Returns the most related document(s) to the query.

2. Similarity search with score  
   Also returns a numerical similarity score.  
   - Lower score → higher similarity  
   - Higher score → less similar  

Similarity search enables answering questions from stored embeddings.

────────────────────────────────────────────────────────
# 9. METADATA FILTERING
────────────────────────────────────────────────────────

Metadata filtering restricts search results to documents that match specific metadata conditions.

Use cases:
- Return only players from a specific IPL team  
- Filter by document type  
- Filter by category or tag  

Filtering dramatically improves accuracy during retrieval.

────────────────────────────────────────────────────────
# 10. UPDATING DOCUMENTS
────────────────────────────────────────────────────────

Updating a document requires:
- The document ID  
- The new document content  
- Metadata (optional)

When a document is updated:
- Chroma re-embeds the new text  
- Replaces the old vector  
- Updates stored metadata  

This allows iterative or real-time content correction in RAG systems.

────────────────────────────────────────────────────────
# 11. DELETING DOCUMENTS
────────────────────────────────────────────────────────

Documents can be deleted by ID.  
Once deleted:
- Embedding is removed  
- Metadata is removed  
- Document no longer appears in search results  

Useful for:
- Data refresh  
- Removing outdated content  
- Dynamic RAG knowledge bases  

────────────────────────────────────────────────────────
# 12. BEST PRACTICES FOR USING VECTOR STORES
────────────────────────────────────────────────────────

1. Always store meaningful metadata.  
2. Choose consistent chunking strategy before embedding.  
3. Avoid mixing different embedding models in the same collection.  
4. Keep chunk sizes around 200–300 tokens for best retrieval performance.  
5. Persist your database so data is not lost.  
6. Use metadata filters for domain-specific queries.  
7. Re-embed documents whenever content changes.  
8. Clean duplicate or low-quality chunks.

────────────────────────────────────────────────────────
# 13. SUMMARY
────────────────────────────────────────────────────────

- A vector database stores embeddings and performs similarity search.  
- A vector store is a LangChain abstraction that interacts with the database.  
- Chroma acts as the actual vector database in this workflow.  
- Documents include both content and metadata.  
- Embeddings allow meaningful similarity comparisons.  
- Chroma supports adding, updating, deleting, and querying documents.  
- Metadata filtering increases retrieval accuracy.  
- Persistence allows reuse across sessions.

This covers the entire vector store and Chroma workflow used in your code.
