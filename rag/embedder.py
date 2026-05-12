"""
rag/embedder.py — Embed chunks using Azure OpenAI and store in VectorStore
"""
from __future__ import annotations
import logging
from typing import List

from core.llm_client import embed_texts
from core.models import Chunk
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


def embed_chunks(chunks: List[Chunk], batch_size: int = 16) -> List[Chunk]:
    """
    Embed all chunks in batches.
    Modifies chunks in-place (sets .embedding) and returns them.
    """
    texts = [c.text for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")

    embeddings = embed_texts(texts)
    assert len(embeddings) == len(chunks), "Embedding count mismatch."

    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb

    logger.info("Embedding complete.")
    return chunks


def build_vector_store(chunks: List[Chunk]) -> VectorStore:
    """Build and return a VectorStore from embedded chunks."""
    store = VectorStore()
    store.build(chunks)
    return store
