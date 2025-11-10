from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# text = """Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

# These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs."""

loader = PyPDFLoader('Langchain\RAG\TEXT SPLITTERS\pdf2.pdf')
splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator= ''
)

docs = loader.load()

result = splitter.split_documents(docs)
print(result[0])