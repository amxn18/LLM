from langchain_text_splitters import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddingModel = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

textSplitter = SemanticChunker(
    embeddings=embeddingModel,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold=3
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.
"""

docs = textSplitter.create_documents([sample])
print(len(docs))
print(docs)
