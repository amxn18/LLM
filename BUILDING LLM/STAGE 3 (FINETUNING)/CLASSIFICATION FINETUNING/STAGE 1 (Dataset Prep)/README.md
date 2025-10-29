# LLM Fine-Tuning: Dataset Preparation for SMS Spam Classification

This README explains step-by-step how the dataset was prepared for fine-tuning a language model (LLM) on a **binary classification task** (spam vs. ham). The process covers **loading, preprocessing, splitting, tokenizing, padding, and batching**.

---

## 1. Loading & Preprocessing the Dataset

- The dataset used is `spam.csv`, which contains SMS messages labeled as **spam** or **ham**.  
- Only the first two columns are required:
  - `v1` → label (`ham` / `spam`)
  - `v2` → SMS text message  

Steps performed:
1. Load dataset with correct encoding (`latin1`).
2. Keep only the required columns (`v1`, `v2`).
3. Balance the dataset by **undersampling ham** so that the number of ham messages equals the number of spam messages.  
4. Shuffle the dataset to ensure randomness.  
5. Convert labels:
   - `ham → 0`
   - `spam → 1`

---

## 2. Splitting the Dataset

The dataset is split into **Training, Validation, and Testing** sets:

- **70% Training** → Model learns from this data.  
- **10% Validation** → Used for **hyperparameter tuning** and monitoring during training. The model never learns from this.  
- **20% Testing** → Used only once after training to evaluate the final model’s performance.  

The split is done randomly with a fixed seed (`random_state=123`) to ensure reproducibility.

---

## 3. Tokenization & Padding

Since the model requires numerical inputs:
1. **Tokenizer**: GPT-2’s tokenizer (`tiktoken`) is used.  
   - Converts each SMS text into a sequence of token IDs.  
2. **Max Length**:
   - If not given, the longest SMS message in the dataset determines the maximum sequence length.  
   - Validation and testing datasets use the same `maxLen` as training.  
3. **Padding**:
   - All sequences are padded to `maxLen` using a special token (`<|endoftext|>`, token ID = `50256`).  
   - Ensures uniform input sizes for the model.

---

## 4. Custom Dataset Class

A `SpamDataset` class (subclass of `torch.utils.data.Dataset`) is created to:
- Load the CSV file.  
- Encode text messages into token IDs.  
- Apply truncation and padding.  
- Return `(encoded_text, label)` as tensors.  

This makes it compatible with PyTorch `DataLoader`.

---

## 5. DataLoaders

Three `DataLoader` objects are created:
- **TrainingLoader** → batches of 8, shuffled, incomplete batches dropped.  
- **ValidationLoader** → batches of 8, no shuffling, all data kept.  
- **TestingLoader** → batches of 8, no shuffling, all data kept.  

Purpose:
- Converts dataset into manageable **mini-batches**.  
- Supports efficient iteration during training & evaluation.  
- Ensures reproducibility with `torch.manual_seed(123)`.

---

##  6. Sanity Check

To confirm everything works:
- Check shapes of input batches (`[batch_size, maxLen]`).  
- Check shapes of target labels (`[batch_size]`).  
- Print total number of batches for train, validation, and test.

---

## Summary of Flow

1. Load raw dataset (`spam.csv`).  
2. Balance dataset (equal ham & spam).  
3. Shuffle and encode labels (`ham=0, spam=1`).  
4. Split into training, validation, testing sets.  
5. Tokenize SMS messages with GPT-2 tokenizer.  
6. Pad/truncate to uniform length.  
7. Wrap in a `SpamDataset` class.  
8. Create DataLoaders for batching and iteration.  

This pipeline ensures the dataset is **clean, balanced, tokenized, padded, and ready for fine-tuning** a classification model.

---
