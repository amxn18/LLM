# Causal Attention Mechanism 

> **Purpose:** Complete conceptual documentation for a Causal (autoregressive) Attention implementation. This README explains the concepts reflected in the provided PyTorch code, including queries, keys, values, masking, softmax, dropout, and context vectors.

---

## 1) Summary 

Causal attention ensures that when computing the representation for a token at position `t`, the model **does not attend to any future tokens (`> t`)**. This is critical for autoregressive tasks like language modeling. In the code, this is achieved via an **upper-triangular mask** applied to the attention scores before the softmax step.

---

## 2) Input and embeddings 

* Inputs are represented as a 2D tensor of shape `(sequence_length, embedding_dim)` for a single example, or `(batch_size, sequence_length, embedding_dim)` for batches.
* Linear layers project input embeddings into **Query (Q), Key (K), and Value (V) matrices**.
* Each linear layer has trainable weights that learn meaningful projections for attention.
* In your code, `nn.Linear` with optional bias is used to implement these projections for Q, K, V.

---

## 3) Computing attention scores 

* Attention scores are computed using the scaled dot product: `scores = Q @ K.T`.
* Scaling by `sqrt(d_k)` stabilizes gradients, especially for higher-dimensional embeddings.
* These scores represent the compatibility between queries and keys before normalization.

---

## 4) Softmax and probability distribution 

* Softmax converts attention scores into attention weights, ensuring each row sums to 1.
* This step allows the model to compute a weighted sum over the values (context vectors).
* In early naive implementations, applying softmax **before masking** can cause leakage of future information. Correct causal attention masks logits before softmax.

---

## 5) Causal masking (upper-triangular mask) 

* Prevents tokens from attending to future positions.
* Implemented as an **upper-triangular matrix** (`torch.triu`) with the diagonal offset by 1.
* Masked positions are set to `-inf` in attention scores before softmax to zero their probability contribution.
* This avoids the “leakage problem” where softmax allows future token influence.

---

## 6) Dropout — regularization and robustness ☔

* Dropout is applied to attention weights after softmax and to the final context vector output.
* Prevents overfitting by randomly zeroing some attention weights during training.
* Ensures that the model does not become overly reliant on any particular token.
* Disabled during evaluation or generation to preserve deterministic outputs.

---

## 7) Context vectors 

* Computed as a weighted sum of value vectors using attention weights: `context = A @ V`.
* Represents the attended information for each token in the sequence.
* In batch mode, this operates over multiple sequences simultaneously.

---

## 8) Handling batches and variable sequence lengths 

* The CausalMechanismV1 class supports batches of sequences `(batch_size, seq_len, dim_in)`.
* Masking is broadcasted across batches, and attention computation respects sequence length.
* Dropout is applied per batch and per position.

---

## 9) Key points reflected in your implementation 

1. **Linear layers** for Q, K, V allow smooth trainable projections.
2. **Scaled dot product** stabilizes training.
3. **Upper-triangular masking** prevents attention to future tokens.
4. **Softmax after masking** ensures proper probability distribution.
5. **Dropout on attention weights** reduces overfitting.
6. **Batch support** allows multiple sequences to be processed simultaneously.
7. **Context vector computation** combines attended information correctly.

---

## 10) Numerical and implementation tips 

* Use `-inf` or large negative values for masked logits.
* Ensure proper broadcasting of masks in batch mode.
* For reproducibility, set random seeds before initializing layers and dropout.
* Validate attention weights sum to 1 after softmax.
* Visualize attention matrices to check causal correctness (lower-triangular after masking).

---

## 11) References 

* "Attention Is All You Need" — Vaswani et al., 2017.
* PyTorch `nn.Linear` and `Dropout` documentation.
* Transformers: GPT family autoregressive attention implementation notes.

---

This README captures all concepts and implementation choices in your provided PyTorch code. It explains **why** each component exists and how causal attention prevents future token leakage while stabilizing training with scaling and dropout.
