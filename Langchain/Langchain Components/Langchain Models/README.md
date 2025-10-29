# LangChain Models Integration – Complete Guide

This document explains how various Large Language Models (LLMs) and Embedding Models are integrated using **LangChain**.  
It covers **Google Gemini**, **Hugging Face Models**, **Open Source Models**, and **Document Similarity Search** using Embeddings.

---

## Setup and Installation

Before integrating models, all core dependencies are installed.  
LangChain acts as a unified framework to connect different model providers (OpenAI, Anthropic, Google, Hugging Face, etc.).

### Key Packages:
- `langchain`, `langchain-core` → Core LangChain libraries.
- `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-huggingface` → Model-specific integrations.
- `transformers`, `huggingface-hub` → For using and loading open-source models locally.
- `numpy`, `scikit-learn` → For embeddings and similarity computations.

---

## Google Gemini Integration (PaLM / Generative AI)

The **Google Gemini (Generative AI)** integration allows direct access to Google’s conversational models.  
Using `ChatGoogleGenerativeAI` from `langchain-google-genai`, a chat-based model can be invoked for text generation.

### Key Points:
- The model used: **`gemini-2.5-flash`**  
- It’s invoked using your **Google API key** stored securely in `.env`.
- Temperature controls the **creativity** of responses (higher = more diverse).

### Flow:
1. The environment is loaded using `dotenv`.  
2. The Gemini model is initialized with `ChatGoogleGenerativeAI`.  
3. A text query (like *“Who is Virat Kohli?”*) is passed for generation.  

This gives a simple conversational output using Google’s LLM through LangChain.

---

## Hugging Face Integrations

LangChain supports **two ways** to connect with Hugging Face models:

### 🔹 A) Using the Hugging Face **API Endpoint**
This approach uses the `HuggingFaceEndpoint` class to connect to a model hosted on Hugging Face’s servers.

**Example Model Used:** `mistralai/Mixtral-8x7B-Instruct-v0.1`  
- Task: Conversational text generation  
- Temperature: Controls randomness in generated responses  
- It creates a `ChatHuggingFace` object for chat-style interaction.

**When to Use:**  
When the model is too large to run locally and needs to be accessed via Hugging Face Inference API.

---

### 🔹 B) Using Local / Open-Source Models

For fully **offline or open-source** operation, the `HuggingFacePipeline` class is used.  
It loads the model directly from the Hugging Face Hub and creates a text-generation pipeline.

**Example Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

**Key Highlights:**
- The model runs **locally** using the `transformers` library.  
- `max_new_tokens` defines the output length.  
- `temperature` defines the randomness in responses.  
- Integrated with `ChatHuggingFace` to enable conversational text generation.

**When to Use:**  
When you want complete control over inference and avoid using APIs.

---

## Embedding Models (Text → Vector Conversion)

Embeddings convert textual information into **numerical vector representations** that capture meaning.  
They are essential for **semantic similarity**, **search**, and **retrieval-based** applications.

### Model Used:
`sentence-transformers/all-MiniLM-L6-v2`  
A lightweight, high-performance embedding model suitable for semantic tasks.

### Two Common Methods:
1. **`embed_query(text)`** → Converts a single query into a 384-dimensional vector.  
2. **`embed_documents(documents)`** → Converts multiple documents into embeddings.

Each vector encodes the semantic meaning of the text so that similar texts produce similar vectors.

---

## Document Similarity Search (Cosine Similarity)

After obtaining embeddings, **cosine similarity** is used to find how close two text vectors are in meaning.

### Concept:
Cosine similarity measures the **angle** between two vectors in multidimensional space.  
A score close to `1` indicates strong similarity, while `0` or negative means dissimilarity.

### Steps:
1. Embed all documents into vector form using the embedding model.  
2. Embed the query (e.g., “Tell me about MS Dhoni”).  
3. Compare query and document embeddings using cosine similarity.  
4. Sort and select the highest similarity score to find the most relevant document.

### Example Understanding:
For a set of cricket-related sentences, if the query mentions *“MS Dhoni”*,  
the document containing details about **MS Dhoni** will yield the highest similarity score.

### Vector Details:
- Default Embedding Dimension: **384**
- Library used: `scikit-learn` (`cosine_similarity` method)

---

## Complete Flow Summary

```text
User Input (Text Query)
       ↓
LLM / Embedding Model (via LangChain Integration)
       ↓
Embedding Vectors or Generated Responses
       ↓
Optional Similarity Search using Cosine Similarity
       ↓
Most Relevant Result or Contextual Response
