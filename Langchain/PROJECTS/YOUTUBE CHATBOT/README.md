# RAG Pipeline from YouTube Transcripts (LangChain + HuggingFace + FAISS)

# 1) INDEXING PIPELINE

The indexing pipeline prepares the raw transcript into a searchable knowledge base.

## 1.1 Document Ingestion
- A YouTube video ID is used to automatically fetch the English transcript.
- The transcript API extracts all caption segments.
- All segments are merged into a single continuous text document.
- Error handling is included for cases where:
  - Transcripts are disabled
  - No captions are available
  - API issues occur

## 1.2 Text Chunking
- The full transcript is split into smaller overlapping chunks.
- Recursive character text splitting is used.
- Chunk size = 400 characters.
- Overlap = 120 characters for better context retention.
- Output is a list of text documents ready for embedding.
- This ensures:
  - Stable retrieval quality
  - Semantic continuity across chunk boundaries

## 1.3 Embedding Generation
- Each chunk is converted into numerical vector embeddings.
- Uses a SentenceTransformer model (MiniLM variant).
- Produces dense 384-dimensional embeddings.
- These embeddings represent semantic meaning of transcript segments.

## 1.4 Vector Store Creation (FAISS)
- All embeddings are stored in a FAISS index.
- FAISS enables fast approximate nearest neighbor (ANN) search.
- Each chunk is mapped to a unique internal ID.
- The vector store supports:
  - Similarity search
  - MMR (Max Marginal Relevance) search
  - Retriever conversion for LangChain chains

---------------------------------------------------------------------

# 2) RETRIEVAL STAGE

## 2.1 Retriever Setup
- The FAISS vector store is transformed into a retriever object.
- Retriever uses MMR search:
  - Ensures diversity in retrieved chunks
  - Reduces redundancy
- Retrieval configuration:
  - Top-k = 4 documents
  - Search type = MMR

## 2.2 Retrieving Context
- A user question is passed into the retriever.
- The retriever returns the most relevant transcript chunks.
- These chunks form the “context” for the LLM.

---------------------------------------------------------------------

# 3) AUGMENTATION STAGE (Context + Query)

## 3.1 LLM Selection
- Uses a HuggingFace endpoint model:
  - Qwen/Qwen3-4B-Instruct-2507
- Generation temperature is lowered (0.3) for factual answers.
- The model is wrapped using LangChain’s Chat interface.

## 3.2 Prompt Construction
- A strict system prompt is used:
  - Answers must come only from transcript context.
  - If context is insufficient, the model must say “I don’t know.”
- The final prompt includes:
  - Retrieved context (all chunk texts)
  - User question
- This ensures grounded answers, not hallucinated information.

---------------------------------------------------------------------

# 4) GENERATION STAGE

## 4.1 Final Answer Pipeline
- The augmented prompt (context + question) is passed to the LLM.
- The model generates a grounded answer based only on the transcript.
- Output is parsed into clean text for display.

---------------------------------------------------------------------

# 5) FULL CHAIN FORMATION (End-to-End RAG)

The system is combined into a single LangChain Runnable pipeline:

1. Input question flows into the retriever.  
2. Retrieved context is formatted into text.  
3. Context and question are passed into the prompt template.  
4. The template goes into the LLM for generation.  
5. Output is parsed and returned as final answer.

This pipeline supports:
- Summarization of the entire transcript
- Query answering grounded in transcript content
- Topic-specific questions (latency, throughput, etc.)
- Safe handling of missing context

---------------------------------------------------------------------

# 6) FEATURES SUPPORTED

- Automatic transcript ingestion from YouTube
- Complete preprocessing pipeline (chunking, embedding, indexing)
- Memory-efficient semantic embeddings
- Fast FAISS vector search
- MMR-based retrieval for diverse results
- LLM answers strictly tied to transcript context
- Fully functional RAG chain for:
  - Question answering
  - Video summarization
  - Topic extraction
  - Clarification questions

---------------------------------------------------------------------

# 7) WHAT THIS RAG SYSTEM DEMONSTRATES

- How to build a RAG pipeline from scratch without frameworks
- How to convert raw data (transcripts) into a searchable knowledge base
- How to retrieve relevant context using FAISS
- How to augment prompts for grounded LLM responses
- How to build a full chain using LangChain Runnables

---------------------------------------------------------------------

# 8) EXTENSIONS YOU CAN ADD LATER

- Add metadata like timestamps for each chunk
- Add multi-video support
- Add better embedding models
- Add reranking
- Build a UI on top of this chain
- Store FAISS index persistently
- Replace HuggingFace endpoint with a local model

---------------------------------------------------------------------
