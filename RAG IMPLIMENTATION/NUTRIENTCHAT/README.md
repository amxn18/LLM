# Local RAG Pipeline: Nutrition Textbook Q&A

## Overview
This repository implements a **framework-free Retrieval-Augmented Generation (RAG)** pipeline for answering questions from a PDF textbook.  
It leverages **semantic search** and **local LLM inference** without using high-level frameworks like LangChain.

* Source PDF: https://pressbooks.oer.hawaii.edu/humannutrition2/

* Colab Notebook Link: https://colab.research.google.com/drive/1EXnOhr8Wc6_tX5P37D6R2A71Y6_hy3mH?usp=sharing

Pipeline Stages:
1. Document Loading & Preprocessing
2. Text Chunking
3. Embedding Generation
4. Vector Search (Semantic Search)
5. Prompt Augmentation
6. LLM Query Answering

---

## 1 Document Loading & Preprocessing
- The PDF textbook is downloaded (if not already present) and opened locally.
- Text is extracted page by page.
- Each page's text is **cleaned** to remove unwanted newlines and spaces.
- Basic statistics are collected for each page: character count, word count, sentence count, and token count.
- Sentences are split using a sentence tokenizer (spaCy) to prepare for chunking.

---

## 2 Text Chunking
- Each page's sentences are split into **smaller chunks** to manage the token limits for embeddings.
- Chunks contain a fixed number of sentences (e.g., 10).
- Very short chunks (below a minimum token threshold) are filtered out to avoid including irrelevant or incomplete content.
- Each chunk is stored along with metadata: page number, character count, word count, and token count.

---

## 3 Embedding Generation
- Each text chunk is converted into a **numerical vector representation** using a sentence embedding model (e.g., SentenceTransformers `all-mpnet-base-v2`).
- The embeddings are stored alongside the corresponding text chunks for later retrieval.
- Embeddings can be saved to a file or database for reuse.

---

## 4 Vector Search (Semantic Search)
- Queries are also converted into embeddings using the same embedding model.
- A **cosine similarity** or **dot product** is computed between the query embedding and all text chunk embeddings.
- The top-k chunks with the highest similarity scores are retrieved as the most relevant context for the query.
- Optional: a reranking model can refine the top results for higher accuracy.

---

## 5 Prompt Augmentation
- The retrieved text chunks are combined into a **context paragraph**.
- A **prompt template** is created, instructing the LLM to answer the query based on the provided context.
- Example references are included in the prompt to guide answer style (few-shot learning).
- This step ensures the LLM leverages retrieved knowledge while maintaining coherent, informative responses.

---

## 6 LLM Query Answering
- A local causal LLM (e.g., Gemma 2B or 7B) is loaded based on available GPU memory.
- The augmented prompt is tokenized and passed to the LLM for generation.
- The model generates a response based on both the query and the retrieved context.
- Output is cleaned to remove the prompt text and any special tokens.
- A helper function (`ask`) wraps this workflow for reusable querying with temperature control, max token limits, and optional context return.

---

## Additional Notes
- **Scalability:** For large datasets (>100k chunks), consider using a vector database (e.g., FAISS) to efficiently store embeddings and perform similarity search.
- **Chunk Size:** Adjust based on the LLM’s token limit to avoid truncation.
- **Prompt Engineering:** Carefully designing the prompt ensures detailed and accurate answers.
- **GPU Requirements:** Model selection depends on available GPU memory; quantization may be necessary for low-memory setups.

---

## Summary
This pipeline demonstrates a fully **local RAG implementation** without relying on frameworks.  
Key features:
- PDF reading and text preprocessing
- Sentence-based chunking with filtering
- Embedding creation for semantic search
- Retrieval of top-k relevant passages
- Prompt augmentation for instruction-following LLMs
- Local LLM inference to generate precise, context-aware answers
