#  Vector Embeddings & Positional Embeddings in NLP

## 🔹 1. What are Token (Vector) Embeddings?
- Words (or tokens) in text are initially represented as **integers (IDs)**.
- These IDs by themselves don’t carry meaning for the model.
- An **Embedding Layer** maps each token ID to a **dense vector** (a list of numbers) that captures semantic meaning.
- For example:
  Token ID → [0.12, -0.35, 0.87, ...]

- These vectors are **learnable parameters**, trained so that similar words get similar embeddings.

 Example:  
If `vocab_size = 50,257` and `vector_dim = 256`,  
the embedding table is a **matrix of shape (50,257 × 256)**.  
Each row corresponds to one token → one embedding vector.

---

## 🔹 2. Why Do We Need Embeddings?
 Convert discrete tokens (integers) into **continuous vectors** that models can process.  
 Capture **semantic meaning** (e.g., "king" and "queen" will be close in vector space).  
 Enable models to generalize better on unseen text.

---

## 🔹 3. Positional Embeddings
- Token embeddings alone **don’t preserve order**.
- Transformers process sequences **in parallel**, so they need extra info about **position**.
- **Positional Embeddings** provide this order information.

---

###  Absolute Positional Embedding
- Each position (0, 1, 2, …, context_length-1) has a unique embedding vector.
- Example:

  Position 0 → [0.11, -0.22, 0.33, ...]  
  Position 1 → [0.45, -0.67, 0.12, ...]

- These are added to the token embeddings:

  Input Embedding = Token Embedding + Positional Embedding

 Shape:  
- Token Embeddings: (Batch_size × Context_len × Vector_dim)  
- Positional Embeddings: (Context_len × Vector_dim)  
- Final Input Embeddings: (Batch_size × Context_len × Vector_dim)

---

###  Relative Positional Embedding
- Instead of encoding **absolute positions**, it encodes **relative distances** between tokens.  
- Example:  
  In the sentence `"I love NLP"`, the model might encode:  
  - Distance between `"I"` and `"love"` is 1  
  - Distance between `"love"` and `"NLP"` is 1  
- This is useful for tasks where **relative position matters more than absolute position**.

---

## 🔹 4. Why Are Positional Embeddings Important?
- Without them, the model sees the input as just a **bag of tokens** with no sense of order.
- Order is crucial in language:
  - `"I love NLP"` ≠ `"NLP love I"`.
- Positional embeddings allow Transformers to understand **sequence structure** while still benefiting from parallelization.

---

## 🔹 5. Final Input Representation
The final embedding fed into the Transformer is:

  Input Embedding = Token Embedding + Positional Embedding

Example with:
- Batch size = 8
- Context length = 4
- Vector dim = 256

 Token embeddings → (8 × 4 × 256)  
 Positional embeddings → (4 × 256)  
 Final input embeddings → (8 × 4 × 256)

---

#  Summary
- **Token Embeddings** → give meaning to tokens.  
- **Positional Embeddings** → give order to tokens.  
- Together, they form the **input embeddings** that Transformers use to learn patterns in text.
