# 🧠 Self-Attention Mechanism (Scaled Dot-Product Attention)

---

## ⚡ Intuition
The **Self-Attention Mechanism** allows each token in a sequence to "look at" other tokens and decide **which words are more important** for understanding its meaning.  
It uses **Query (Q)**, **Key (K)**, and **Value (V)** matrices to calculate weighted representations of tokens.

---

## 🪜 Step-by-Step Breakdown

### 🔹 S1) Converting Input into Query, Key, Value Matrices
- Each input embedding is projected into 3 different spaces:
  - **Q = X × Wq** → Query Matrix
  - **K = X × Wk** → Key Matrix
  - **V = X × Wv** → Value Matrix  
- Here, `Wq`, `Wk`, and `Wv` are **trainable weight matrices** that get updated during training.
- Intuition:
  - **Query (Q):** What this token is looking for.
  - **Key (K):** What this token contains.
  - **Value (V):** The actual information to pass along.

---

### 🔹 S2) Attention Score Matrix
- To measure **how much attention one token should give to others**, we compute:
  - For example, for token **x² (journey)**:
    - w21 = q2 ⋅ k1  
    - w22 = q2 ⋅ k2  
    - w23 = q2 ⋅ k3  
- This gives us a **matrix of attention scores**:  AttentionScores = Q × K^T

---

### 🔹 S3) Scaling + Softmax Normalization
- Problem: Dot products can get **large** when `dim(K)` is high → softmax becomes **peaky** (one token dominates).
- Solution: **Scale by √(dim(K))**

* ScaledScores = AttentionScores / √(dim(K))
* Then apply **Softmax** along the last dimension: AttentionWeights = softmax(ScaledScores, dim=-1)

- ✅ Why scaling?
1. Prevents **training instability**.  
2. Keeps **variance stable** (avoids exploding values).  
3. Ensures smooth probability distribution.  

---

### 🔹 S4) Computing Context Vectors
- Final output for each token is a **weighted sum of Value vectors (V)**: Context(i) = Σ ( AttentionWeights(i, j) × Value(j) )

* In matrix form: ContextMatrix = AttentionWeights × V
- This produces **context-aware representations**, where each token embedding contains information from other relevant tokens.

---

## 🔑 What Happens During Training?
- **Wq, Wk, Wv are trainable**.  
- During backpropagation:
- Gradients update these matrices.  
- Model learns **how queries match keys** and **what values to propagate**.  
- Over time, the model learns meaningful attention patterns (e.g., in language: subject ↔ verb connections).  

---

## 🧩 Why Q, K, V Split?
- If we used only one matrix, each token would attend to itself in a trivial way.  
- Splitting into **Q, K, V** allows:
- Queries → express what the token "wants".  
- Keys → provide what each token "offers".  
- Values → carry the actual information.  
- This separation makes the mechanism **flexible** and **powerful**.  

---

## 🏗️ Implementations in Code
- **Version 1 (Manual Weights):**
- Uses `nn.Parameter` for Wq, Wk, Wv.  
- Computes attention step-by-step.  
- **Version 2 (Better Initialization):**
- Uses `nn.Linear` layers for Wq, Wk, Wv.  
- More stable, smoother training due to better initialization schemes.  

---

##  Final Intuition
- Self-Attention answers:  
- **“Which tokens should I focus on to understand the current token?”**  
- Context vectors → enriched embeddings used in Transformers.  
- This is the foundation of models like **BERT, GPT, Vision Transformers, etc.**

---