# Pretraining a Mini GPT Model from Scratch

This project implements a **miniature GPT-like model** trained on a book dataset.  
It demonstrates how modern LLMs (like GPT-2/3/4/ChatGPT) are pretrained,  
covering the entire pipeline from tokenization to training and evaluation.

---

## 1. Dataset and Tokenization
- The input is a raw text file (`the-verdict.txt`).
- A **Byte Pair Encoding (BPE)** tokenizer (`tiktoken`, GPT-2 style) is used.
- Each word/character is mapped into **tokens** (integers).
- The dataset is split into **training (90%)** and **validation (10%)**.
- A **sliding window approach** is used to create overlapping sequences:
  - Input: `x1, x2, ..., xn`
  - Target: `x2, x3, ..., xn+1`  
  (next-token prediction).

---

##  2. Model Architecture
The model is a **Transformer-based GPT** with the following components:

### 🔹 Token & Positional Embeddings
- Words (tokens) → converted into **dense vectors**.
- Positional embeddings preserve word order in sequences.

### 🔹 Transformer Blocks
Each block consists of:
1. **Layer Normalization** – stabilizes training by normalizing inputs.
2. **Masked Multi-Head Self-Attention**  
   - Queries, Keys, Values are computed.  
   - Masking ensures the model **only looks at past tokens** (causality).  
   - Multi-heads capture multiple types of relationships.
3. **Residual Connections + Dropout** – improves gradient flow and prevents overfitting.
4. **Feed-Forward Network (FFN)** – expands and compresses embeddings for richer representations.

### 🔹 Output Layer
- Final layer maps embeddings → **vocabulary logits**.
- Softmax over logits → probability distribution for next token.

---

##  3. Training Process
- **Loss Function**: Cross-Entropy Loss (predicting next token).
- **Optimizer**: AdamW (weight decay improves generalization).
- **Training Loop**:
  1. For each batch:
     - Forward pass → predictions.
     - Compute loss → compare predictions with targets.
     - Backpropagation → update weights.
  2. After every `evalFreq` steps:
     - Evaluate on both training & validation sets.
     - Track loss curves.

- **Overfitting Check**:  
  - If validation loss ≫ training loss → model is overfitting.

---

##  4. Evaluation & Sample Generation
- After each epoch, the model generates text using:
  1. Start context prompt → `"Every effort moves you"`.
  2. Greedy decoding (argmax selection).
  3. New tokens appended sequentially.
- This simulates how GPT continues text generation.

---

##  5. Loss Visualization
- Loss curves are plotted against:
  - **Epochs** → overall training progress.
  - **Tokens seen** → how much data the model has processed.
- Helps identify:
  - Underfitting (high losses).
  - Overfitting (validation loss much higher than training loss).

---

## 6. Key Insights
- The project mimics **GPT pretraining** on a smaller scale:
  - **Causal language modeling** (predict next token).
  - **Transformer architecture** (attention + residuals + FFN).
  - **Training/validation monitoring**.
- Unlike real GPT models:
  - Dataset here is **one book** (vs trillions of tokens).
  - Model has **124M parameters** (vs billions in GPT-3).
  - Training is on a local GPU (vs massive clusters).

---

##  7. Real-World Connection
- This pipeline mirrors how OpenAI and others train LLMs.
- Steps like **tokenization, embeddings, attention, loss, optimization**  
  are **identical in principle** to full-scale GPT pretraining.
- Scaling up:
  - More data (internet-scale).
  - Bigger models (billions of parameters).
  - Longer training (weeks/months on clusters).
