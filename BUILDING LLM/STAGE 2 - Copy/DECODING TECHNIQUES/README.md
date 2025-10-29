# 🚀 Language Model Training, Evaluation & Decoding Strategies  

This project implements a **GPT-like Transformer model** from scratch in PyTorch and demonstrates **training, evaluation, and decoding strategies**.  

---

##  1. Model Components  

### 🔹 Layer Normalization  
- Normalizes each token’s embedding to stabilize training.  
- Uses two trainable parameters:  
  - **Scale (γ)** → adjusts magnitude  
  - **Shift (β)** → re-centers distribution  
- Helps reduce **internal covariate shift** and speeds up convergence.  

---

### 🔹 GELU Activation  
- **Gaussian Error Linear Unit (GELU)** is used instead of ReLU.  
- Provides **smooth non-linearity**, letting the model capture subtle relationships.  
- Works better than ReLU/Tanh in large-scale Transformers.  

---

### 🔹 Feed-Forward Network (FFN)  
- Each Transformer block has an **expansion → non-linearity → compression** flow.  
- Expansion: Embedding dimension → 4x larger.  
- Activation: GELU applied.  
- Compression: Projected back to original embedding size.  
- Adds **capacity** for richer feature representation.  

---

### 🔹 Multi-Head Self Attention (MHSA)  
- Core mechanism of Transformers.  
- Splits embeddings into multiple **heads** to attend to different parts of context.  
- **Causal Masking** ensures tokens cannot “see the future”.  
- Produces **context vectors** that capture relationships across tokens.  

---

### 🔹 Transformer Block  
- Combines **Multi-Head Attention + FFN** with:  
  - **Layer Normalization**  
  - **Residual (Skip) Connections**  
  - **Dropout for regularization**  
- Stacking multiple blocks deepens model capacity.  

---

### 🔹 GPT Model Architecture  
- Components:  
  1. **Token Embeddings** → convert IDs to vectors.  
  2. **Positional Embeddings** → inject sequence order info.  
  3. **Stacked Transformer Blocks** → core computation.  
  4. **Final Linear Head** → maps embeddings back to vocabulary logits.  

---

##  2. Dataset Preparation  

- Raw text is tokenized into **GPT-2 tokens**.  
- A **sliding window** approach creates overlapping input-target pairs.  
- **Training set** → 90% of data  
- **Validation set** → 10% of data  
- Each batch provides input tokens (X) and target tokens (Y).  

---

##  3. Training & Evaluation  

- **Loss Function** → Cross Entropy (compares predicted logits with true tokens).  
- **Optimizer** → AdamW (weight decay for regularization).  
- **Overfitting Check**:  
  - Training loss << Validation loss → Overfitting.  
- **Metrics**:  
  - Training Loss  
  - Validation Loss  
  - Tokens Seen (progress measure)  

- Visualization: Loss curves vs **epochs** and **tokens seen**.  

---

## 4. Decoding Strategies  

Once trained, the model generates text using **decoding strategies**.  

### 🔹 A) Greedy Decoding  
- Always selects the token with **highest probability**.  
- Deterministic → same output every time.  
- Limitation → **Repetitive & dull outputs**.  

---

### 🔹 B) Temperature Scaling (🌡️ Hyperparameter)  
- Controls **randomness** in sampling.  
- Formula → `Softmax(logits / T)`  
- Effects of Temperature (T):  
  - **T → 0** → Sharp distribution → behaves like greedy decoding.  
  - **T = 1** → Normal probabilities (default).  
  - **T > 1** → Flatter distribution → more diverse but less accurate outputs.  
-  Balances **determinism vs creativity**.  

---

### 🔹 C) Top-K Sampling  
- Restricts sampling to **top K most probable tokens**.  
- Remaining logits replaced with `-∞` (ignored).  
- Ensures:  
  - No sampling from very low-probability words.  
  - Improves coherence while keeping diversity.  

---

### 🔹 D) Combined (Top-K + Temperature)  
- Typical modern strategy:  
  1. **Filter tokens using Top-K**.  
  2. **Rescale with Temperature**.  
  3. **Sample from multinomial distribution**.  
-  Ensures **coherence** (via Top-K) and **creativity** (via Temperature).  

---

##  5. Key Insights  

1. **Training Loss vs Validation Loss** → indicates generalization ability.  
2. **Decoding Strategy** drastically affects **quality of text**.  
3. **Temperature (T)** is a **hyperparameter** that controls creativity.  
4. **Top-K filtering** avoids random, nonsensical tokens.  
5. Final pipeline:  Logits → Top-K filtering → Temperature scaling → Softmax → Multinomial sampling

