# GPT-2 SMS Spam Classifier 

1) Dataset Preparation

Dataset Source and Structure:
- SMS messages labeled as spam or ham (not spam).
- Columns:
  - Label: spam or ham
  - Message: SMS text

Balancing the Dataset:
- Original dataset is imbalanced (more ham than spam).
- Balanced dataset created by:
  - Counting spam messages.
  - Randomly sampling equal number of ham messages.
  - Combining for equal class representation.

Splitting the Dataset:
- Training Set (70%): Model learning.
- Validation Set (10%): Hyperparameter tuning, generalization check.
- Testing Set (20%): Final evaluation.

Conceptual Note:
- Validation prevents overfitting by monitoring performance on unseen data.

2) Data Tokenization and DataLoader Creation

Tokenization:
- SMS texts converted into token IDs using GPT-2 tokenizer.
- Sequences padded to uniform length.

Dataset Class:
- Loads CSV data.
- Encodes messages to token IDs.
- Pads sequences to max length.
- Returns tokenized input and labels.

DataLoader:
- Creates batches for efficient GPU processing.
- Training batches shuffled to prevent order bias.
- Validation/testing batches sequential.

Conceptual Note:
- Batching & padding enable parallel computation and maintain context across sequences.

3) Model Architecture

Base GPT-2 Model:
- Transformer-based language model predicting next token.
- Pretrained on large corpora capturing syntactic, semantic, and contextual relationships.

Key Components:
- Token Embeddings: Map token IDs to high-dimensional vectors.
- Positional Embeddings: Encode token positions.
- Transformer Blocks:
  - Multi-Head Self-Attention: Attend to all tokens simultaneously.
  - Feed-Forward Layers: Complex transformations.
  - Layer Normalization: Stabilizes training.
  - Residual Connections: Preserve info and prevent vanishing gradients.
- Output Head: Predicts vocabulary distribution (pretraining).

Activation & Normalization:
- GELU: Smooth nonlinear activation.
- LayerNorm: Feature-wise normalization for numerical stability.

4) Modifying GPT-2 for Classification

Classification Head:
- Original output head replaced with linear layer mapping embedding → 2 classes.
- Only last token output used for sequence-level classification.

Freezing Pretrained Layers:
- All GPT-2 layers frozen except:
  - Last transformer block
  - Final LayerNorm
  - Classification head

Conceptual Note:
- Transfer learning retains linguistic features.
- Reduces training time and prevents overfitting.

5) Finetuning the Model

Training Steps:
- Forward Pass: Input → GPT-2 → last-token logits.
- Loss Computation: Cross-entropy vs true labels.
- Backward Pass: Gradients computed for trainable params.
- Optimizer Step: AdamW updates weights.
- Evaluation: Periodic loss/accuracy checks.

Evaluation Metrics:
- Loss: How predictions match labels; lower is better.
- Accuracy: Fraction of correct predictions.

Visualization:
- Loss & accuracy curves plotted vs epochs & examples seen.
- Identifies underfitting, overfitting, and training stability.

6) Using the Finetuned Model

Inference Pipeline:
- Preprocessing: Truncate/pad input to max length.
- Tokenization: Text → token IDs.
- Batch Dimension: Added for transformer input.
- Model Prediction: Forward pass (no gradients).
- Last Token Extraction: Used for final class prediction.
- Class Mapping: Argmax → SPAM / NOT SPAM.

Example Usage:
- Promotional SMS → SPAM
- Personal SMS → NOT SPAM
- Practical spam detection in real-world messages.

7) Key Learnings

- Transfer Learning in NLP: Pretrained models accelerate task-specific training.
- Attention Mechanisms: Capture dependencies efficiently.
- Sequence-Level Classification: Last token summarizes input.
- Model Freezing: Prevents catastrophic forgetting, reduces compute.
- Evaluation Strategy: Balanced dataset + validation prevents bias.

8) Project Impact

- Real-time spam detection in messaging platforms.
- Demonstrates GPT-2 adaptability beyond text generation.
- Template for finetuning GPT-2 for other classification tasks.

9) Next Steps / Extensions

- Experiment with GPT variants (medium, large) for higher accuracy.
- Introduce data augmentation for robustness.
- Explore multi-class classification beyond spam/ham.
- Integrate into messaging apps or automated filtering systems.
