# Cross-Entropy Loss & Perplexity in Language Models

## 🔹 What is Cross-Entropy Loss?
Cross-entropy loss is the **most commonly used loss function for training language models**.  
It measures how different the predicted probability distribution is from the actual distribution.

- The model outputs **logits** → raw scores (not probabilities).
- We apply **softmax** to convert logits into probabilities across the vocabulary.
- For each token, cross-entropy picks the probability of the **correct target word**.
- We take the **negative log** of these probabilities and average them.

The goal of training is to **minimize this loss**, so the model assigns higher probability to the correct words.

---

## 🔹 How Cross-Entropy Works (Step by Step)

1. **Input & Target Sequences**
   - Input: `"Every effort moves"`
   - Target: `"effort moves you"`

   The model tries to predict the next word at each position.

2. **Model Output (logits)**
   - Shape: `(batch_size, num_tokens, vocab_size)`
   - Example: `(2, 3, 50257)`

3. **Softmax Probabilities**
   - Converts logits into probabilities for each token in the vocabulary.

4. **Target Probabilities**
   - Extract the probabilities assigned to the actual target tokens.

---

## 🔹 Two Implementations in Code

1. **Manual Computation**
   ```python
   target_probs = probs[text_idx, [0,1,2], targets[text_idx]]
   log_probs = torch.log(target_probs)
   avg_log_probs = log_probs.mean()
   neg_avg_log_probs = -avg_log_probs

* Why Negative Log?

Probabilities range between 0 and 1.

Log of small numbers (like 0.0001) becomes large negative values.

Taking the negative flips it into a positive error signal.

Minimizing this ensures probabilities for the correct tokens approach 1

 ## Perplexity (PPL)

Perplexity is derived from cross-entropy loss and is a key evaluation metric in NLP.

--> Formula:
Perplexity = e^(CrossEntropyLoss)

* Intuition:
It measures how "surprised" the model is by the correct data.

* Lower perplexity = better model.

* Example:
Cross-entropy = 2.3 → Perplexity ≈ 10  
Cross-entropy = 1.0 → Perplexity ≈ 2.7  

* If perplexity = k, it means the model is as uncertain as if it had to uniformly guess among k words.

