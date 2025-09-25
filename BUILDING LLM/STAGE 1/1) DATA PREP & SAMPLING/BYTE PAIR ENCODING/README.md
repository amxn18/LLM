# 🔹 Byte Pair Encoding (BPE) in Tokenization

## What is BPE?
Byte Pair Encoding (BPE) is a subword tokenization algorithm widely used in modern NLP and LLMs.  
It was originally a text compression method, but later adapted for tokenization to overcome the limitations of **word-level** and **character-level** tokenization.

---

## Why not Word-level or Character-level Tokenization?
- **Word-level tokenization**  
  - Huge vocabulary size (every word is unique).  
  - Struggles with out-of-vocabulary (OOV) words.  
  - Memory heavy and inefficient.  

- **Character-level tokenization**  
  - Very small vocabulary (only alphabets, digits, symbols).  
  - Sequences become too long.  
  - Harder for model to capture semantic meaning.  

- **BPE strikes a balance:**  
  - Keeps frequent words intact.  
  - Splits rare/unknown words into smaller subwords.  
  - Reduces vocabulary size.  
  - Preserves root words (e.g., "play", "playing", "played").  

---

## How BPE Works (Step-by-Step)
1. **Start with character-level encoding**  
   Example: `"low"` → `["l", "o", "w", </w>]`  
   - `</w>` marks the end of a word.  

2. **Count symbol pairs**  
   - Find the most frequent adjacent pair in the corpus.  

3. **Merge most frequent pair**  
   - Replace that pair with a new symbol.  
   - Example: `("l", "o")` → `"lo"`.  

4. **Repeat for fixed iterations (or until vocab limit reached)**  
   - Each iteration merges the most frequent pair.  

5. **Two key rules in BPE**  
   - Keep **frequent words** as they are.  
   - Break **rare words** into subwords.  

---

##  Vocabulary Reduction
- Starts with a **small initial vocabulary** (characters).  
- Iteratively merges pairs → vocabulary grows but **much smaller than full word vocab**.  
- Example: Instead of having `"unhappiness"`, `"happiness"`, `"happily"` separately,  
  - Store `"hap"`, `"ness"`, `"ily"` once.  
  - Reuse subwords across multiple words.  

This reduces size while still capturing root forms.

---

## How to Decide Iterations?
- The number of merges (iterations) is usually a **hyperparameter**.  
- Too few merges → results in character-level encoding.  
- Too many merges → similar to word-level encoding.  
- In practice, iteration count is tuned so the final vocabulary size is within a target (e.g., 30k–50k tokens).  

---

## BPE in GPT Models
- **GPT-1 & GPT-2** used BPE as their primary tokenization technique.  
- Later models (GPT-3, GPT-4) evolved further with optimized tokenization strategies.  
- `tiktoken` is the open-source library (by OpenAI) that implements efficient tokenization (inspired by BPE + optimizations).  

---

## tiktoken Library
- A fast and memory-efficient tokenizer library from OpenAI.  
- Used in GPT models for splitting text into tokens.  
- Provides pretrained vocabularies compatible with GPT-2, GPT-3, GPT-4.  
- Ensures consistency between tokenization and model training.  
- Example usage:
  ```python
  import tiktoken
  enc = tiktoken.get_encoding("gpt2")
  tokens = enc.encode("Hello world")
  print(tokens)        # → [15496, 995]
  print(enc.decode(tokens))  # → "Hello world"
