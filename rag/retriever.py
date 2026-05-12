"""
rag/retriever.py — Multi-factor ranked retrieval with graph expansion
"""
from __future__ import annotations
import logging
from typing import List

from core.models import Chunk, RetrievalResult
from core.vector_store import VectorStore
from core.llm_client import embed_texts
from rag.graph import ChunkGraph
from config.settings import (
    TOP_K_RETRIEVAL,
    GRAPH_EXPANSION_DEPTH,
    WEIGHT_SIMILARITY,
    WEIGHT_SECTION,
    WEIGHT_GRAPH,
    WEIGHT_POSITION,
)

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, vector_store: VectorStore, graph: ChunkGraph):
        self.vector_store = vector_store
        self.graph = graph

    # ── Main retrieve ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[RetrievalResult]:
        """
        Full retrieval pipeline:
        1. Embed query
        2. FAISS top-k search
        3. Multi-factor ranking
        4. Graph expansion
        """
        # Step 1 — Embed query
        query_embedding = embed_texts([query])[0]

        # Step 2 — Vector search
        raw_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        if not raw_results:
            logger.warning("No results from vector search.")
            return []

        # Identify query section hints (keywords from query)
        query_keywords = set(query.lower().split())

        # Step 3 — Multi-factor scoring
        max_position = max((c.position for c, _ in raw_results), default=1) or 1
        ranked: List[RetrievalResult] = []

        for chunk, sim_score in raw_results:
            section_score = self._section_relevance(chunk, query_keywords)
            graph_score = self.graph.connectivity_score(chunk.id)
            position_score = 1.0 - (chunk.position / max_position)

            final_score = (
                WEIGHT_SIMILARITY * sim_score
                + WEIGHT_SECTION * section_score
                + WEIGHT_GRAPH * graph_score
                + WEIGHT_POSITION * position_score
            )

            ranked.append(
                RetrievalResult(
                    chunk=chunk,
                    similarity_score=sim_score,
                    section_score=section_score,
                    graph_score=graph_score,
                    position_score=position_score,
                    final_score=final_score,
                )
            )

        ranked.sort(key=lambda r: r.final_score, reverse=True)
        top_results = ranked[:top_k]

        # Step 4 — Graph expansion
        top_ids = [r.chunk.id for r in top_results]
        expanded_chunks = self.graph.expand(top_ids, depth=GRAPH_EXPANSION_DEPTH)

        # Add expanded chunks with reduced scores
        existing_ids = set(top_ids)
        for ec in expanded_chunks:
            if ec.id not in existing_ids:
                top_results.append(
                    RetrievalResult(
                        chunk=ec,
                        similarity_score=0.0,
                        section_score=0.0,
                        graph_score=self.graph.connectivity_score(ec.id),
                        position_score=0.0,
                        final_score=0.1,   # graph-expanded bonus
                    )
                )
                existing_ids.add(ec.id)

        logger.info(f"Retrieved {len(top_results)} chunks ({len(top_results) - len(top_ids)} graph-expanded)")
        return top_results

    # ── Scoring helpers ────────────────────────────────────────────────────────

    def _section_relevance(self, chunk: Chunk, query_keywords: set) -> float:
        """Check if section title overlaps with query keywords."""
        if not chunk.section:
            return 0.0
        section_words = set(chunk.section.lower().split())
        overlap = len(section_words & query_keywords)
        return min(1.0, overlap / 3.0)
