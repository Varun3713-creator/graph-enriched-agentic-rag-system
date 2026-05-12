"""
core/vector_store.py — FAISS index persistence and search
"""
from __future__ import annotations
import os
import json
import pickle
import logging
from typing import List, Tuple

import numpy as np
import faiss

from core.models import Chunk
from config.settings import FAISS_INDEX_DIR

logger = logging.getLogger(__name__)

INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(FAISS_INDEX_DIR, "chunks.pkl")


class VectorStore:
    """Wraps a FAISS flat-L2 index with chunk metadata storage."""

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None  # Inner-product (cosine via normalized vecs)
        self.chunks: List[Chunk] = []
        self.dim: int = 0

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: List[Chunk]) -> None:
        """Build a new FAISS index from embedded chunks."""
        assert all(c.embedding for c in chunks), "All chunks must have embeddings before building."
        vectors = np.array([c.embedding for c in chunks], dtype="float32")
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)
        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.chunks = chunks
        logger.info(f"FAISS index built with {len(chunks)} vectors (dim={self.dim})")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Return top-k (chunk, score) pairs for a query embedding."""
        if self.index is None or self.index.ntotal == 0:
            return []
        vec = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))
        return results

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self) -> None:
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_FILE)
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Vector store saved to {FAISS_INDEX_DIR}")

    def load(self) -> bool:
        """Load existing index. Returns True if successful."""
        if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
            return False
        self.index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        self.dim = self.index.d
        logger.info(f"Vector store loaded: {len(self.chunks)} chunks")
        return True

    def __len__(self) -> int:
        return len(self.chunks)
