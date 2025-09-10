#  Multi-Head Attention Mechanism (MHA)

##  What is Multi-Head Attention?
Multi-Head Attention is an extension of the **Self-Attention mechanism** where instead of having just a single "attention head", we use **multiple attention heads**.  
Each head learns different representation subspaces, allowing the model to **capture diverse relationships** within the input sequence.

---

##  Key Concepts

### 1️ Self-Attention Recap
- Each token is projected into **Query (Q), Key (K), and Value (V)** vectors.
- Attention weights are computed as: Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

- Captures **dependencies between tokens** in the sequence.

---

### 2️ Why Multi-Head?
- A single self-attention head may only focus on limited relationships.
- Multiple heads = multiple "views" of the sequence.
- Each head learns **different attention patterns** in parallel.
- Outputs from all heads are **concatenated and linearly projected** back.

---

### 3️ Two Implementations in Code
#### a) **Naive Multi-Head (Wrapper Approach)**
- Multiple independent self-attention layers are created.  
- Their outputs are **concatenated along the last dimension**.  
- Limitation: parameters grow linearly with the number of heads.

#### b) **Optimized Multi-Head (Weight Splitting Approach)**
- Instead of creating separate layers, we use a **single projection layer** for Q, K, and V.  
- Then we **reshape and split** the vectors into multiple heads.  
- More efficient and closer to how Transformers actually implement MHA.

---

##  Causal Masking
- Used to ensure **no token can attend to future tokens** (important in autoregressive models like GPT).  
- Achieved by filling future positions in the attention score matrix with `-∞`.  
- This enforces **causal/left-to-right attention**.

---

##  Workflow of Multi-Head Attention
1. **Linear Projections** → Input embeddings are projected into Q, K, V spaces.
2. **Head Splitting** → Each Q, K, V is split into multiple heads.
3. **Scaled Dot-Product Attention** → Attention weights are computed per head.
4. **Masking** → Apply causal mask to block future information (if needed).
5. **Concatenation** → Combine outputs from all heads.
6. **Final Projection** → Pass concatenated result through a linear layer.

---

##  Benefits of Multi-Head Attention
- Captures **different types of dependencies** (short-term & long-term).  
- Allows the model to attend to information at **multiple representation subspaces**.  
- Improves expressiveness and generalization of the Transformer architecture.  

---

##  Where is it used?
-  Transformers (BERT, GPT, T5, etc.)  
-  Encoder-Decoder architectures  
-  NLP tasks (translation, summarization, Q&A)  
-  Vision Transformers (ViTs) in computer vision  

---

## ⏱ Complexity
- **Time Complexity:** `O(n² · d)` (due to QKᵀ computation across tokens).  
- **Space Complexity:** `O(n²)` for storing attention weights.  

---

#  Summary
Multi-Head Attention = Multiple parallel self-attention heads → concatenated & projected → richer contextual representations.  
It is the **core building block of Transformers** and enables them to understand sequences effectively.  

