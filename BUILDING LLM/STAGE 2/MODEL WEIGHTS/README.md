# PRETAINED BASE GPT MODEL 

## High-level Workflow (from raw text to trained model)
1. Load raw text file(s).  
2. Tokenize with GPT-2 BPE tokenizer (`tiktoken`) to get token IDs.  
3. Create sliding-window input→target pairs (Dataset) and wrap with PyTorch DataLoader.  
4. Build a decoder-only Transformer (token embeddings + positional embeddings → stacked Transformer blocks → final layernorm → linear head to logits).  
5. Train with causal language modeling objective (predict next token) using cross-entropy loss.  
6. Monitor training and validation loss; generate text samples periodically.  
7. Save checkpoints and, if desired, map pretrained GPT-2 weights into your model for warm-start/fine-tuning.

---

## Detailed Explanations

### 1) Tokenization & Encoding
- Purpose: convert raw unicode text into a sequence of integer token IDs.
- Approach used: GPT-2 Byte Pair Encoding (BPE) via `tiktoken`. BPE produces subword tokens so unknown words are decomposed into known subword units instead of `<unk>`.
- Practical notes:
  - Use `encoding = tiktoken.get_encoding("gpt2")`.
  - When tokenizing entire corpora, consider allowed special tokens like `<|endoftext|>` if you want explicit document boundaries.
  - Keep track of total tokens and vocabulary size for model config.

### 2) Input–Target Pair Creation (Sliding Window)
- Goal: create training examples suitable for autoregressive next-token prediction.
- For context length L (max_len):
  - Input X = tokens[i : i+L]
  - Target Y = tokens[i+1 : i+1+L]
- Stride controls overlap:
  - stride = L → non-overlapping chunks (faster, fewer samples)
  - stride < L → overlapping chunks (more examples, more training signal)
- Implementation detail:
  - Store inputChunks and targetChunks as tensors and return them in `__getitem__`.
  - Choose `drop_last=True` during DataLoader for consistent batch shapes or handle ragged last batch in validation.

### 3) Dataset Split (Train / Validation)
- Basic split: e.g., 90% train / 10% validation by token count or by document split.
- For time-series-like corpora (books), contiguous split can leak structure; consider shuffling or splitting by documents.
- Sanity checks:
  - Ensure `len(train_tokens) >= context_length`.
  - Compute total training/validation tokens (#samples × context_length) for logging.

### 4) Embeddings
- Token Embeddings:
  - A learnable matrix E ∈ R^(V × d_model). Lookup E[token_id] → (d_model).
  - Created with `nn.Embedding(vocab_size, d_model)`; learned from scratch or initialized from pretrained weights.
- Positional Embeddings:
  - Two common options:
    - Learnable absolute positional embeddings (matrix of shape (max_pos, d_model)).
    - Rotary or relative encodings for improved long-range generalization (advanced).
  - Combined by addition: input_embedding = token_embedding + positional_embedding.
- Scaling:
  - Some implementations multiply token embeddings by sqrt(d_model) for numerical stability (optional).

### 5) Transformer Block (Decoder-only)
- Core components per block:
  1. LayerNorm
  2. Causal Multi-Head Self-Attention (MHA)
     - Queries, Keys, Values linear projections.
     - Split into H heads, each head dim = d_model / H.
     - Scaled dot-product attention: softmax(QK^T / sqrt(head_dim)) · V.
     - Apply causal mask: prevent tokens from attending to future positions (strictly upper-triangular masked).
  3. Residual connection + Dropout
  4. Feed-Forward Network (FFN)
     - Two linear layers with expansion factor (typically 4x) and nonlinearity (GELU).
  5. Final residual + layernorm
- Implementation notes:
  - Use `register_buffer` for fixed causal mask for efficiency.
  - Ensure proper reshaping/transposing for (batch, heads, seq_len, head_dim) operations.
  - Keep `qkv_bias` option consistent when mapping pretrained weights.

### 6) Output Head & Weight Tying
- Final layer: project model hidden states (d_model) to vocab logits with a linear layer (d_model → V).
- Weight tying (optional): share token embedding matrix and output head weights (transpose). This reduces parameters and often improves performance.

### 7) Loss Calculation (Causal LM)
- Use CrossEntropy over flattened logits vs flattened target tokens.
- Typical implementation:
  - Flatten logits from shape (batch, seq_len, V) to (batch*seq_len, V)
  - Flatten targets to (batch*seq_len)
  - `F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_id)` if padding exists.
- When evaluating per-batch loss for logging, average across batches.

### 8) Training Loop & Evaluation
- Typical steps per batch:
  1. `optimizer.zero_grad()`
  2. forward → logits
  3. compute loss
  4. `loss.backward()`
  5. `optimizer.step()`
- Periodic evaluation:
  - Switch to `model.eval()` and `torch.no_grad()` for loss on a subset (`evalIter` batches) for train/validation metrics.
  - Track and log trainingLoss and validationLoss arrays.
- Early warning signs:
  - Validation loss >> training loss → overfitting.
  - Loss not decreasing → learning rate issues, data problems, or bugs.

### 9) Sampling / Decoding Strategies
- Greedy (argmax): deterministic and fast; often repetitive.
- Temperature scaling: divide logits by temperature before softmax.
  - Temperature < 1.0 → sharper distribution (more deterministic).
  - Temperature > 1.0 → flatter distribution (more varied).
- Top-k sampling: restrict candidates to top k logits before sampling.
- Top-p (nucleus) sampling: choose smallest set of tokens whose cumulative probability ≥ p.
- Combine temperature + top-k/top-p for controlled creativity.
- Implement stop condition with `eosId` to end generation early.

### 10) Checkpointing & Saving/Loading
- Save `state_dict()` of the model and optimizer in a single checkpoint:
  - Save training step / epoch / best validation loss to resume cleanly.
- When loading:
  - Recreate model/optimizer and call `load_state_dict`.
  - If resuming training, also load optimizer state.
- Keep periodic checkpoints and one "best" checkpoint (lowest val loss).

### 11) Mapping Pretrained GPT-2 Weights (Conceptual)
- Pretrained weights are often provided in frameworks with different shapes/layouts; mapping requires:
  1. Identifying corresponding parameters (token embeddings `wte`, positional `wpe`, layer norms, attention projections, MLP weights).
  2. QKV layout: some checkpoints store QKV as one concatenated matrix (c_attn with 3×d) — split it into Q, K, V in the same order as the implementation.
  3. Transpose differences: TensorFlow checkpoints may need transposes to match PyTorch layout.
  4. Bias handling: ensure biases are assigned and shapes match.
  5. Validate with shape checks and a small forward pass after load (sanity-check generation).
- Always write a helper `assign()` that checks shapes and raises clear error messages on mismatch.

---

## Hyperparameters (Recommended starting values)
- vocab_size: 50257 (GPT-2 tokenizer)
- context_length: 256 (local); 1024 for larger experiments if resources allow
- d_model (emb_dim): 768 (124M model), 1024, 1280, 1600 (scaled models)
- n_layers: 12
- n_heads: 12 (d_model must be divisible by n_heads)
- dropout: 0.1
- batch_size: 2–8 (depending on GPU memory)
- learning_rate: 1e-4 to 5e-4 (use AdamW)
- weight_decay: 0.1 (tune)
- epochs: depends on data; log tokens seen as a progress measure
- stride: 128 or context_length (controls overlap)

---

## Practical Tips & Debugging Checklist
- Shape sanity checks:
  - After each linear and reshape, assert expected shapes: (batch, seq, d_model), (batch, heads, seq, head_dim), etc.
- NaN / Inf checks:
  - Watch for exploding loss; clip gradients or reduce LR if necessary.
- Validation > Training loss:
  - Try increasing regularization (dropout, weight decay), use more training data, reduce model capacity, or improve data split.
- Training too slow / out of memory:
  - Reduce batch size, lower context_length, use gradient accumulation, or enable mixed precision (AMP).
- Randomness & reproducibility:
  - Set seeds (`torch.manual_seed`), but expect nondeterminism when using multiple workers or some CUDA ops.
- Logging:
  - Track global step, tokens seen, training and validation loss arrays, and sample generations.

---

## Performance & Scaling Advice
- To scale beyond local GPU constraints:
  - Use gradient accumulation to simulate larger batch sizes.
  - Use `torch.cuda.amp` (automatic mixed precision) to halve memory usage.
  - For multi-GPU: use DataParallel or DistributedDataParallel (DDP) / FSDP for larger models.
- For faster retrieval during generation of long contexts, consider Rotary Positional Embeddings (RoPE) or relative position biases.

---

## How to run (conceptual sequence)
1. Prepare environment (install PyTorch and tiktoken).
2. Place raw text in `data/the-verdict.txt`.
3. Run preprocessing script to tokenize and create datasets (or let Dataset handle tokenization on the fly).
4. Configure `GPT_CONFIG` hyperparameters.
5. Start training script; monitor logs and sample outputs periodically.
6. After training or when resuming, load checkpoint and run sample generation or evaluation.

---

## Experiments to Run (examples)
- Change stride to 128 vs context_length to study data redundancy effect.
- Compare learnable positional embeddings vs sinusoidal for local dataset.
- Test temperature and top-k combinations for diversity control.
- Freeze token embeddings and fine-tune only transformer layers (cost-saving experiment).
- Warm start from GPT-2 weights vs training from scratch: compare convergence speed and final loss.

---

## Reproducibility & Best Practices
- Log random seeds and configuration for each run.
- Save checkpoints with training metadata (epoch, global_step, lr, best_val).
- Keep a small validation set aside and never touch it while tweaking hyperparameters.
- Use a consistent tokenizer and vocab for training, evaluation, and generation.

---

## Common Pitfalls & How to Avoid Them
- Using an incorrect causal mask orientation → leads to leakage from future tokens. Verify mask is strictly upper triangular masked.
- Mismatched QKV splitting when loading pretrained checkpoints → always verify order and shapes.
- Forgetting to call `model.eval()` during evaluation/generation → leads to dropout during eval and inconsistent metrics.
- Not using `torch.no_grad()` during evaluation/generation → increased memory use and slower execution.
- Saving only model weights but not optimizer state → cannot resume optimizer momentum state.

---

## Hardware & Runtime Tips
- On a single GPU (e.g., T4 16GB), use context_length = 256 and batch_size small (1–4) for d_model~768 models.
- For larger context/ models, use GPUs with larger memory or split across multiple GPUs.
- Use Colab Pro / TPU for experimentation but beware of library compatibility with tiktoken and PyTorch on TPU.

---

## Final Notes (mapping pretrained GPT-2 weights)
- Mapping weights is delicate: implement and test a `load_weights` helper that:
  1. Prints expected vs provided shapes.
  2. Splits concatenated QKV matrices as required.
  3. Applies transposes if necessary.
  4. Assigns param-by-param and runs a short forward pass to check for runtime errors.
- Keep an untrained model checkpoint to compare outputs for debugging.

---

## Suggested Reading / Next Steps
- Revisit "Attention is All You Need" for the math of scaled-dot product attention.
- Read GPT/GPT-2 papers for architectural choices and training practices.
- After this implementation, move to:
  - Pretraining on larger corpora or fine-tuning
  - Retrieval-Augmented Generation (RAG) and LangChain integration
  - Deployment and inference optimizations (quantization, distillation)

---

## Quick Checklist Before Training (copy & use)
- [ ] Tokenizer installed and working (`tiktoken`)  
- [ ] `the-verdict.txt` in data/ and total tokens computed  
- [ ] Dataset sliding-window parameters (context_length, stride) chosen  
- [ ] Model config (vocab_size, d_model, n_layers, n_heads) set and consistent  
- [ ] Attention causal mask shape verified  
- [ ] Loss and optimizer set (AdamW) and learning rate chosen  
- [ ] Device availability (CUDA) checked and `model.to(device)` used  
- [ ] Checkpoint save path configured

