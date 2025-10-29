OBJECTIVE  
We are creating Input–Target pairs from raw text using a sliding window approach.  
These pairs are then fed into PyTorch’s Dataset & DataLoader classes, forming the backbone of training for language models like GPT.

1️ TOKENIZATION (BPE with TikToken)
   - The raw text is read and tokenized using Byte Pair Encoding (BPE).
   - Each word or subword is mapped into a unique token ID.
   - Example:
     "The cat sat" → [1212, 356, 987]

2️ CREATING INPUT–TARGET PAIRS
   - Using a context size (e.g., 4), we create training samples:
     Input (X): [1, 2, 3, 4]  
     Target (Y): [2, 3, 4, 5]
   - This shifting mechanism makes the model learn next-token prediction.

3️ SLIDING WINDOW APPROACH
   - Instead of creating pairs one by one, we slide across the text.
   - Each step produces overlapping sequences of input–target pairs.
   - Parameters:
     - max_length → window size for each input chunk
     - stride → step size to move the window

4️ DATASET CLASS
   - Custom PyTorch `Dataset` stores all (input, target) pairs.
   - `__getitem__`: returns the pair at a specific index.
   - `__len__`: total number of pairs.

5️ DATALOADER
   - Wraps the Dataset into iterable batches for training.
   - Key parameters:
     - batch_size → number of samples per batch
     - shuffle → randomize order each epoch
     - drop_last → drop incomplete batch
     - num_workers → parallel workers for speed

6️ BATCHING EFFECT
   - Small batch size → faster updates, but noisy gradients.
   - Large batch size → stable gradients, but slower updates.

Raw Text  
   ↓ (Tokenization - BPE)  
Token IDs  
   ↓ (Sliding Window)  
Input–Target Pairs  
   ↓ (Dataset Class)  
PyTorch Dataset  
   ↓ (DataLoader)  
Batches → Model Training

 Input–Target shifting is the core of next-token prediction.  
 Sliding windows ensure overlapping context for better learning.  
 Dataset + DataLoader make training scalable and efficient.  
 Batch size selection impacts both training speed and stability.  
