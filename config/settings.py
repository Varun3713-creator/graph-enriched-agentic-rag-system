"""
config/settings.py — Central configuration loaded from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Azure OpenAI ──────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT: str = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY: str = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
)

# ── Chunking ──────────────────────────────────────────────────
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
SLIDING_WINDOW_SIZE: int = int(os.getenv("SLIDING_WINDOW_SIZE", "400"))

# ── Retrieval ─────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
GRAPH_EXPANSION_DEPTH: int = int(os.getenv("GRAPH_EXPANSION_DEPTH", "2"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# ── Ranking weights ───────────────────────────────────────────
WEIGHT_SIMILARITY: float = 0.5
WEIGHT_SECTION: float = 0.2
WEIGHT_GRAPH: float = 0.2
WEIGHT_POSITION: float = 0.1

# ── Paths ─────────────────────────────────────────────────────
DOCS_DIR: str = os.getenv("DOCS_DIR", "./data/docs")
FAISS_INDEX_DIR: str = os.getenv("FAISS_INDEX_DIR", "./data/faiss_index")
