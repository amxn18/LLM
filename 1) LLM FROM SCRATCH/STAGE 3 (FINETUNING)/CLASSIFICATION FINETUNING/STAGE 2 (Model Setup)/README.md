# GPT-2 Fine-Tuning for SMS Spam Classification

## Project Overview

This project fine-tunes a pretrained GPT-2 model for binary
classification of SMS messages into 'spam' or 'ham'.

GPT-2 is originally a language model; here, it is adapted
for classification by adding a final output head and
training on a labeled SMS dataset.

## Dataset Preparation

- Load the SMS dataset containing text messages and labels.
- Balance the dataset to ensure equal representation of 'spam'
  and 'ham' to prevent model bias.
- Convert labels into numerical format:
  'ham' → 0, 'spam' → 1
- Split dataset into:
  • Training set (70%)
  • Validation set (10%) – used for hyperparameter tuning
  • Test set (20%) – used for final evaluation

## DataLoader & Tokenization

- Tokenize all messages using GPT-2 tokenizer.
- Pad or truncate sequences to uniform length matching model's
  context window.
- Create PyTorch Dataset objects for training, validation, and
  testing.
- Use DataLoaders for batching, shuffling, and efficient
  training.

## Model Setup

- Load pretrained GPT-2 weights (e.g., GPT-2 small, 124M).
- Add a classification head for binary output.
- Freeze all pretrained layers except:
  • Last transformer block
  • Final layer normalization
  • Classification head
- Fine-tune only task-specific parts while retaining general
  language knowledge.

--------------------------------------------------------
## Training & Fine-Tuning
--------------------------------------------------------
- Forward pass: convert input text to token IDs and feed into GPT-2.
- Compute classification loss from final token output.
- Backpropagate gradients only through trainable layers.
- Hyperparameters include learning rate, batch size, epochs,
  and dropout rate.

## Evaluation

- Evaluate performance on test set.
- Metrics: Accuracy, Precision, Recall, F1-Score
- Extract predictions using the final token to determine 'spam'
  or 'ham'.

## Key Concepts Highlighted

- Balanced Dataset: prevents bias towards majority class
- Tokenization & Padding: converts variable-length text into
  fixed-size sequences
- Freezing Layers: reduces computation and preserves pretrained
  knowledge
- Classification Head: adapts GPT-2 for supervised tasks
- Transformer Block Fine-Tuning: trains only last block to
  specialize on spam detection
