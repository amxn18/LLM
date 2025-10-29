# Simplified Attention Mechanism (Without Trainable Weights)

This project implements a **basic attention mechanism** using PyTorch, but without any trainable parameters.  
The goal is to understand *how attention works under the hood* before adding complexity like queries, keys, and values with learned weights.

---

##  Core Idea

Attention allows the model to decide **which tokens in a sequence are most relevant to the current token**.  
It works by assigning weights (attention scores) to different tokens, and then combining them into a **context vector**.

---

##  Steps in Simplified Attention

### 1️ Input Representations
- Each word/token is represented as a **fixed embedding vector** in 3D space.  
- The sequence of words forms a matrix of shape **(number of tokens × embedding dimension)**.

---

### 2️ Attention Scores
- Choose a **query token** (e.g., *"journey"*).  
- Compute the **dot product** between this token’s vector and every other token vector.  
- This gives a **raw attention score** that measures similarity/relevance.  

 Higher dot product = tokens are more related.

---

### 3 Normalization (Attention Weights)
- Raw scores are not interpretable, so they are normalized into probabilities:  
  - **Divide by sum of scores** (basic normalization), or  
  - **Softmax function** (preferred, stable, ensures all weights are positive and sum to 1).  
- Now each token has a weight that represents its importance relative to the query.

---

### 4️ Context Vector
- Multiply each token embedding by its corresponding **attention weight**.  
- Sum them all up → this gives a **context vector** for the query token.  
- The context vector is a weighted representation of the whole sequence, focusing on relevant tokens.

---

### 5️ Extending to All Tokens
- Instead of focusing on just one query token, repeat the above process for every token.  
- This produces:  
  - An **attention matrix** (scores between every pair of tokens).  
  - A **context matrix** (context vectors for each token).  

👉 In practice, we compute this efficiently with matrix multiplications instead of loops.

---

##  Summary

1. **Compute Attention Scores** → similarity via dot products.  
2. **Normalize Scores** → use softmax to get attention weights.  
3. **Compute Context Vectors** → weighted sum of embeddings.  
4. **Generalize to All Tokens** → build full attention and context matrices.  

---

## 🎯 Key Takeaways
- Attention ≠ complicated magic — it’s **just dot products, softmax, and weighted sums**.  
- Without trainable weights, attention still captures meaningful relationships between tokens.  
- The next step in building real transformers is adding **learned projections (queries, keys, values)**, which make attention adaptive.  

---
