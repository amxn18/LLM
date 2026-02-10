#  Model Evaluation: Training and Validation Loss in GPT

This README explains **how training and validation loss are calculated** for a custom GPT-like model.  
We will walk step by step through dataset preparation, dataloaders, splitting data, calculating loss, and interpreting results.  

---

##  1. Dataset Preparation
- The model is trained on a text file (`the-verdict.txt`).  
- Raw text is **tokenized** using the GPT-2 tokenizer (`tiktoken`), which converts text into numerical token IDs.  
- Example: `"Every effort moves you"` → `[502, 1234, 923, ...]`.  

---

##  2. Dataset Class & Dataloader
- A custom **`GPTDatasetV1`** class is created.  
- It works by using a **sliding window** over the tokenized text:
  - Input chunk = `[0 → maxLen]`  
  - Target chunk = `[1 → maxLen+1]`  
- This prepares pairs of input (X) and output (Y) sequences.  

- **Dataloader**:
  - Groups multiple input-target pairs into batches.  
  - Makes training efficient with parallel loading.  

---

##  3. Splitting Data
- Dataset is split into **training** and **validation** sets.  
- `trainRatio = 0.90` →  
  - 90% of tokens used for training  
  - 10% of tokens used for validation  
- Ensures the model is trained on one portion while performance is tested on unseen data.  

---

##  4. Training & Validation Tokens
- Total tokens are counted separately:  
  - **Training Tokens** → used for updating weights.  
  - **Validation Tokens** → used only for evaluation (no weight updates).  
- Helps check if dataset size is sufficient for both splits.  

---

## 5. Cross Entropy Loss
- The loss function used is **Cross-Entropy Loss**.  
- Purpose: measure the difference between predicted probability distribution and the actual target distribution.  
- Formula:  
  \[
  Loss = -\frac{1}{N} \sum (y_{true} \cdot \log(y_{pred}))
  \]  

- Implementation:  
  - Model outputs **logits** (unnormalized predictions).  
  - Logits are reshaped and compared with targets.  
  - Loss ignores padding tokens (set with `ignore_index=0`).  

---

## 6. Loss Calculation
- **`calculateLossBatch`** → Calculates loss for one batch.  
- **`calculateLossLoader`** → Loops through the entire dataloader and computes average loss.  

- Results:
  - **Training Loss** → Measures how well the model fits the training data.  
  - **Validation Loss** → Measures how well the model generalizes to unseen data.  

---

## 7. Device Optimization
- Runs on **GPU (CUDA)** if available, else CPU.  
- GPU (like NVIDIA T4 in Colab) speeds up forward passes and loss computation.  

---

## 8. Interpreting Training vs Validation Loss
- **Training Loss ↓** → Model is learning patterns from training data.  
- **Validation Loss ↓** → Model generalizes well on unseen data.  
- **If Validation Loss >> Training Loss** → Overfitting (model memorizing training data).  
- **If Both Losses are high** → Underfitting (model not learning enough).  

---

## 9. Why Loss Matters
- Loss is the **primary metric** to monitor model quality.  
- Guides:
  - Model performance during training.  
  - Decisions like early stopping, tuning hyperparameters, or adjusting model size.  

---

## 10. Key Takeaways
Dataset is tokenized and prepared using a sliding window.  
Dataloader batches data for efficiency.  
Dataset is split into training (90%) and validation (10%).  
Cross-Entropy Loss is used for evaluation.  
Training Loss → Fit quality, Validation Loss → Generalization quality.  
Comparing both losses detects **overfitting/underfitting**.  

---

# Final Note
This step sets the foundation for **training GPT models**.  
Monitoring **training vs validation loss** is the most critical step before moving to **optimizer setup, backpropagation, and parameter updates**.
