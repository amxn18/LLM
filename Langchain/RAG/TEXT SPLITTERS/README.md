# TEXT SPLITTERS 

This document explains the four major types of text splitters used in LangChain for RAG systems:
1. Character-based splitting  
2. Text-based (recursive) splitting  
3. Document-structured splitting (language-aware)  
4. Semantic-meaning–based splitting  

It also covers why splitting is required, how each splitter behaves, best practices, and when to use each one.

────────────────────────────────────────────────────────
# 1. WHAT ARE TEXT SPLITTERS?
────────────────────────────────────────────────────────

Text splitters break large text content into smaller, well-formed chunks.  
This is essential for RAG because:

- LLMs have limited context windows  
- Embedding models perform best on moderate-sized text  
- Vector search retrieves more accurate matches when chunks are coherent  
- Smaller chunks reduce cost, memory, and latency  
- Retrieval quality improves when each chunk contains a single idea  

Good chunking directly improves the quality of your RAG answers.

────────────────────────────────────────────────────────
# 2. CHARACTER-BASED SPLITTING
────────────────────────────────────────────────────────

This approach splits text strictly by a fixed number of characters.  
It does not consider:
- Sentence boundaries  
- Word boundaries  
- Topic changes  

Splitting is rigid and mechanical.

Characteristics:
- Produces equally sized chunks  
- Fastest and simplest method  
- Useful for baseline testing or synthetic datasets  

Limitations:
- Breaks sentences in the middle  
- Poor semantic coherence  
- Not ideal for retrieval unless used for controlled experiments  

Use when you need strict chunk sizes, not meaningful chunks.

────────────────────────────────────────────────────────
# 3. TEXT-BASED SPLITTING (RECURSIVE CHARACTER SPLITTER)
────────────────────────────────────────────────────────

This is the most commonly used splitter in RAG pipelines.  
It attempts to break text at natural boundaries using a fallback hierarchy:
1. Paragraph separators  
2. Sentence boundaries  
3. Word boundaries  
4. Characters  

If text cannot be split cleanly at a higher level, it falls back to finer granularity.

Characteristics:
- Produces semantically clean, coherent chunks  
- Works well for articles, blogs, reports, documentation, PDFs, and webpages  
- Balanced between readability and chunk size  
- Most reliable for dense retrieval systems  

This is the **default recommended splitter** for general RAG use.

────────────────────────────────────────────────────────
# 4. DOCUMENT-STRUCTURED (LANGUAGE-AWARE) SPLITTING
────────────────────────────────────────────────────────

This splitter is designed for structured formats such as:
- Markdown  
- Python code  
- HTML  
- JSON  
- Notebooks  

It understands syntax and formatting rules of the target language.  
It avoids breaking text in ways that disrupt structure.

Examples:
- Markdown headings are kept intact  
- Code blocks stay grouped  
- Class definitions aren’t split mid-function  
- Lists, tables, and sections remain coherent  

Best for:
- README files  
- Code repositories  
- Technical documentation  
- Structured content with hierarchy or markup  

It preserves the logic and structure of documents better than normal text-based splitters.

────────────────────────────────────────────────────────
# 5. SEMANTIC-MEANING–BASED SPLITTING
────────────────────────────────────────────────────────

This is the most advanced splitter.  
Instead of using characters or punctuation, it uses **embeddings** to detect meaning shifts.

How it works:
- Converts sentences into embeddings  
- Measures changes in semantic similarity  
- Places a split when meaning shifts significantly  

For example:
- Text about farming followed by text about cricket → separate chunks  
- Paragraphs with consistent meaning → combined into bigger chunks  

Characteristics:
- Most accurate chunking  
- Produces topic-aware sections  
- Improves retrieval quality drastically  
- Best for long documents containing multiple subjects  

Trade-offs:
- Slightly slower  
- Uses embedding model  
- More computationally expensive  

Ideal for high-quality RAG systems where accuracy matters more than speed.

────────────────────────────────────────────────────────
# 6. COMPARISON TABLE
────────────────────────────────────────────────────────

| Splitter Type                   | What It Splits By               | Best Use Case                         | Strength                    | Weakness                         |
|---------------------------------|----------------------------------|----------------------------------------|-----------------------------|----------------------------------|
| Character-based                 | Fixed number of characters       | Testing, rigid chunking                | Fast, simple                | Breaks sentences, low coherence  |
| Text-based (recursive)          | Paragraph → sentence → word      | Most RAG pipelines                     | Natural, reliable chunks    | Not structure-aware              |
| Document-structured             | Syntax / markup / code blocks    | Code, Markdown, technical docs         | Preserves structure         | Not meaning-based                |
| Semantic-meaning-based          | Embedding similarity             | High-quality retrieval across topics   | Best accuracy               | Slower, embedding cost           |

────────────────────────────────────────────────────────
# 7. BEST PRACTICES FOR SPLITTING IN RAG
────────────────────────────────────────────────────────

1. Use recursive text splitting for general-purpose RAG.  
2. Use semantic splitting when text contains multiple unrelated topics.  
3. Use language-aware splitting when working with:
   - Markdown  
   - Python code  
   - HTML  
   - Developer documentation  
4. Avoid pure character splitting except for experiments.  
5. Use chunk overlap (10–20%) to maintain context continuity.  
6. Recommended chunk sizes:
   - 200–300 tokens for embedding-based retrieval  
   - Larger for summarization tasks  
7. Always balance chunk size with model context limits.  
8. Choose splitter based on:
   - Content structure  
   - Retrieval quality requirements  
   - Speed and cost constraints  

────────────────────────────────────────────────────────
# 8. SUMMARY
────────────────────────────────────────────────────────

- Character-based splitting is rigid and mechanical.  
- Recursive text splitting is the most practical and balanced option.  
- Document-structured splitting is ideal for code and markdown.  
- Semantic chunking produces the most meaningful and accurate RAG retrieval.  

Each splitter serves a specific need.  
Choosing the correct one increases retrieval accuracy, reduces cost, and improves overall LLM performance.
