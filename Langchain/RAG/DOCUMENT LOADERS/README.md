# RAG DOCUMENT LOADERS 

- Documentation - https://docs.langchain.com/oss/python/integrations/document_loaders
────────────────────────────────────────────────────────
# 1. WHAT ARE DOCUMENT LOADERS?
────────────────────────────────────────────────────────

Document Loaders are components that import data from different sources into LangChain.  
Each loader:
- Reads external data (text, pdf, html, csv, folders, websites)
- Converts them into a list of Document objects
- Each Document contains:
  - page_content → the text extracted
  - metadata → page number, source path, URL, etc.

Document loaders are foundational for RAG pipelines because LLMs need clean text to work with.

────────────────────────────────────────────────────────
# 2. TEXTLOADER
────────────────────────────────────────────────────────

Used to load plain `.txt` files.

CODE:
from langchain_community.document_loaders import TextLoader

loader = TextLoader('Langchain/RAG/DOCUMENT LOADERS/cricket.txt', encoding='utf-8')
docs = loader.load()
docs[0].page_content

EXPLANATION:
- Reads raw text files
- Returns a list of Document objects
- Ideal for poems, articles, notes, e-books, or any plain text

When used with runnables:
loaderChain = RunnableLambda(lambda _: loader.load()[0].page_content)

This allows fully automatic loading inside a chain:
load -> prompt -> model -> parser

────────────────────────────────────────────────────────
# 3. PYPDFLOADER
────────────────────────────────────────────────────────

Used to extract text from PDF files.

CODE:
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("pdf2.pdf")
docs = loader.load()

EXPLANATION:
- Splits PDF into pages
- Each `Document` represents a single page
- docs[0].page_content contains page 1 text

Features:
- Handles multi-page documents
- Works for scanned PDFs only when OCR tools are used (not included)

────────────────────────────────────────────────────────
# 4. WEBBASELOADER
────────────────────────────────────────────────────────

Used to scrape text directly from any public webpage.

CODE:
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(url)
docs = loader.load()

EXPLANATION:
- Fetches HTML content
- Extracts readable text using BeautifulSoup
- Returns Document objects containing webpage content

Example pipeline using runnables:
- Auto-load webpage
- Auto-generate question
- Feed both to prompt
- Pass to LLM

This enables webpage-based QA without manually copy-pasting content.

────────────────────────────────────────────────────────
# 5. CSVLOADER
────────────────────────────────────────────────────────

Used to load CSV files row-by-row.

CODE:
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Social_Network_Ads.csv')
data = loader.load()

EXPLANATION:
- Reads each CSV row into a Document
- page_content holds a string representation of the row
- metadata contains row index and file source

Best for:
- Structured datasets
- Tabular RAG pipelines
- Querying ads, sales data, accounts, user info, etc.

────────────────────────────────────────────────────────
# 6. DIRECTORYLOADER
────────────────────────────────────────────────────────

Loads an entire folder of documents in one go.

CODE:
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("path/to/folder", glob="*.txt")
docs = loader.load()

EXPLANATION:
- Recursively scans folder
- Loads all matching documents
- Supports PDF, text, markdown, CSV, HTML, etc.

Useful for:
- Chat with your files
- Large RAG databases
- Document ingestion pipelines

────────────────────────────────────────────────────────
# 7. DIFFERENCE BETWEEN load() AND lazy_load()
────────────────────────────────────────────────────────

All loaders provide:

### load()
- Eager execution
- Loads entire dataset into memory
- Returns a list immediately
- Best for small/medium-sized files

### lazy_load()
- Generator-based streaming
- Loads documents one at a time
- More memory-efficient
- Best for large PDFs, huge directories, or long web pages

Example:

docs = loader.load()
vs.

for doc in loader.lazy_load():
    process(doc)

Lazy loading prevents memory overload and improves performance for large datasets.

────────────────────────────────────────────────────────
# 8. WHEN TO USE WHICH LOADER?
────────────────────────────────────────────────────────

| Loader          | Best For                           | Extracts           |
|-----------------|-------------------------------------|--------------------|
| TextLoader      | Notes, poems, articles, books       | Raw text           |
| PyPDFLoader     | Reports, e-books, research papers   | Page-wise text     |
| WebBaseLoader   | Product pages, blogs, docs          | HTML → clean text  |
| CSVLoader       | Tables, datasets, logs              | Row-wise text      |
| DirectoryLoader | Entire folders, multi-type data     | Bulk load          |

────────────────────────────────────────────────────────
# 9. SUMMARY
────────────────────────────────────────────────────────

- TextLoader: Use for plain text files.
- PyPDFLoader: Extracts PDF pages as documents.
- WebBaseLoader: Scrapes webpage content.
- CSVLoader: Loads rows from datasets.
- DirectoryLoader: Bulk ingestion of file folders.
- load(): immediate extraction.
- lazy_load(): stream extraction for large files.

These loaders are essential for any RAG pipeline, allowing transformations such as:
- Splitting
- Embedding
- Storing in vector databases
- Retrieval
- Passing into LLMs for contextual answers

