# Workflow of GPT Model
Example Input: "Every Effort Moves You"

--------------------------------------------------
Step 1: Input Sequence
--------------------------------------------------
- Raw sequence is taken as input.
- Example: "Every Effort Moves You"

--------------------------------------------------
Step 2: Tokenization
--------------------------------------------------
- Sequence is split into tokens.
- Each token is mapped to a unique Token ID from the vocabulary.
  Example:
  "Every" → 2013, "Effort" → 4509, "Moves" → 7231, "You" → 4321

--------------------------------------------------
Step 3: Token Embeddings
--------------------------------------------------
- Each Token ID is converted into a dense vector representation.
- For GPT-2, embedding dimension (emb_dim) = 768.
- Embeddings are initialized randomly and optimized during training.

--------------------------------------------------
Step 4: Positional Embeddings
--------------------------------------------------
- Captures the position of each token in the sequence.
- Ensures order information is preserved.
- Dimension = 768 (same as token embeddings).

--------------------------------------------------
Step 5: Input Embeddings
--------------------------------------------------
- Token Embeddings + Positional Embeddings = Input Embeddings.
- Dimension remains 768.
- These form the input to the Transformer blocks.

--------------------------------------------------
Step 6: Dropout Layer
--------------------------------------------------
- Applied on Input Embeddings before entering Transformer block.
- Randomly sets some values to 0 based on dropout rate.
- Advantages:
  1) Prevents overfitting
  2) Improves generalization

--------------------------------------------------
Transformer Block (×12 in GPT-2 small)
--------------------------------------------------
Purpose: Builds contextual relationships between tokens.
Input: Input embeddings (dim = 768) for each token.

--------------------------------------------------
Step 7.1: Layer Normalization 1
--------------------------------------------------
- Normalizes token embeddings.
- Sets mean = 0 and variance = 1.
- Advantages:
  1) Speeds up convergence
  2) Improves stability and efficiency

--------------------------------------------------
Step 7.2: Masked Multi-Head Attention
--------------------------------------------------
- Core of Transformer.
- Each token embedding attends to other tokens (with masking for autoregression).
- Produces Context Vectors:
  - Contain semantic meaning
  - Capture relationships between tokens
  - Learn importance/weightage of each token

--------------------------------------------------
Step 7.3: Dropout
--------------------------------------------------
- Dropout applied again after attention mechanism.
- Helps with regularization.

--------------------------------------------------
Step 7.4: Residual Connection
--------------------------------------------------
- Adds original input (from before attention) to the attention output.
- Prevents vanishing gradients and preserves information.

--------------------------------------------------
Step 7.5: Layer Normalization 2
--------------------------------------------------
- Normalizes the residual output for stability.

--------------------------------------------------
Step 7.6: Position-wise Feed Forward Network (FFN)
--------------------------------------------------
- Fully connected network applied independently to each token embedding.
- Captures non-linear transformations and enriches representations.

--------------------------------------------------
Step 7.7: Dropout + Residual Connection
--------------------------------------------------
- Dropout applied to FFN output.
- Added back to the input (residual).

--------------------------------------------------
Step 8: Final LayerNorm + Output Layer
--------------------------------------------------
- After passing through all Transformer blocks, final Layer Normalization is applied.
- Output dimension: (sequence_length × 768)
  Example: For "Every Effort Moves You" → 4 × 768

- This output is passed through a linear layer (Output Head).
- The linear layer projects embeddings to vocabulary size (50257 for GPT-2).
  Example: 4 × 768 → 4 × 50257

- Each row now represents a probability distribution across the vocabulary.
- Softmax is applied to convert logits into probabilities.

- The token with the highest probability is selected as the next predicted token.

--------------------------------------------------
Next Word Prediction (Autoregressive Generation)
--------------------------------------------------
S1) Extract the last vector (corresponding to the last token).  
S2) Apply SOFTMAX to convert logits into probabilities that sum to 1.  
S3) Identify the index position (Token ID) of the largest probability value and decode this Token ID.  
S4) Append the predicted Token ID to the previous input sequence for the next word prediction.  

Note: If the model is not trained yet, it will produce random Token IDs since the weights are not optimized.
