"""
rag/graph.py — Graph-based chunk linking using NetworkX
"""
from __future__ import annotations
import logging
import math
from typing import List, Set

import networkx as nx
import numpy as np

from core.models import Chunk, ChunkLink, LinkType
from config.settings import GRAPH_EXPANSION_DEPTH

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.82   # cosine sim threshold for SIMILAR links
ENTITY_OVERLAP_THRESHOLD = 2  # min shared keywords for ENTITY_MATCH


class ChunkGraph:
    """
    Directed graph of chunk relationships.
    Nodes: chunk IDs
    Edges: typed links (NEXT, PARENT, SIMILAR, ENTITY_MATCH)
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._chunk_map: dict[str, Chunk] = {}

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: List[Chunk]) -> None:
        """Construct graph from a list of chunks."""
        self.graph.clear()
        self._chunk_map = {c.id: c for c in chunks}

        # Add all nodes
        for chunk in chunks:
            self.graph.add_node(chunk.id, section=chunk.section, position=chunk.position)

        # NEXT links: sequential order
        for i in range(len(chunks) - 1):
            self._add_link(chunks[i].id, chunks[i + 1].id, LinkType.NEXT, weight=1.0)

        # PARENT links: same section → heading → content
        section_heading: dict[str, str] = {}
        for chunk in chunks:
            from core.models import ChunkType
            if chunk.chunk_type == ChunkType.HEADING:
                section_heading[chunk.section] = chunk.id
            elif chunk.section in section_heading:
                self._add_link(section_heading[chunk.section], chunk.id, LinkType.PARENT, weight=0.9)

        # SIMILAR links: cosine similarity between embeddings
        self._add_similarity_links(chunks)

        # ENTITY_MATCH links: shared keyword overlap
        self._add_entity_links(chunks)

        logger.info(
            f"Graph built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def _add_link(self, src: str, tgt: str, link_type: LinkType, weight: float = 1.0) -> None:
        self.graph.add_edge(src, tgt, link_type=link_type.value, weight=weight)

    def _add_similarity_links(self, chunks: List[Chunk]) -> None:
        embedded = [c for c in chunks if c.embedding]
        if len(embedded) < 2:
            return
        vectors = np.array([c.embedding for c in embedded], dtype="float32")
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = vectors / norms
        sim_matrix = normed @ normed.T

        for i in range(len(embedded)):
            for j in range(i + 1, len(embedded)):
                sim = float(sim_matrix[i, j])
                if sim >= SIMILARITY_THRESHOLD and embedded[i].section != embedded[j].section:
                    self._add_link(embedded[i].id, embedded[j].id, LinkType.SIMILAR, weight=sim)
                    self._add_link(embedded[j].id, embedded[i].id, LinkType.SIMILAR, weight=sim)

    def _add_entity_links(self, chunks: List[Chunk]) -> None:
        """Link chunks that share significant keyword overlap."""
        STOPWORDS = {"the", "a", "an", "is", "in", "of", "and", "to", "for", "with", "that", "this", "are", "be"}

        def keywords(text: str) -> Set[str]:
            words = text.lower().split()
            return {w.strip(".,;:!?\"'()") for w in words if len(w) > 3 and w not in STOPWORDS}

        chunk_keywords = [(c, keywords(c.text)) for c in chunks]

        for i in range(len(chunk_keywords)):
            for j in range(i + 1, len(chunk_keywords)):
                ci, ki = chunk_keywords[i]
                cj, kj = chunk_keywords[j]
                overlap = len(ki & kj)
                if overlap >= ENTITY_OVERLAP_THRESHOLD and ci.id != cj.id:
                    weight = min(1.0, overlap / 10.0)
                    self._add_link(ci.id, cj.id, LinkType.ENTITY_MATCH, weight=weight)

    # ── Expand ────────────────────────────────────────────────────────────────

    def expand(self, chunk_ids: List[str], depth: int = GRAPH_EXPANSION_DEPTH) -> List[Chunk]:
        """
        Expand a set of chunk IDs by following graph edges up to `depth` hops.
        Returns additional chunks (not in original set).
        """
        visited: Set[str] = set(chunk_ids)
        frontier = set(chunk_ids)

        for _ in range(depth):
            new_frontier: Set[str] = set()
            for node in frontier:
                for neighbor in self.graph.successors(node):
                    if neighbor not in visited:
                        new_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = new_frontier

        expanded = [self._chunk_map[cid] for cid in visited if cid not in set(chunk_ids) and cid in self._chunk_map]
        return expanded

    def connectivity_score(self, chunk_id: str) -> float:
        """Return a normalized connectivity score for a chunk (0–1)."""
        if chunk_id not in self.graph:
            return 0.0
        degree = self.graph.degree(chunk_id)
        max_degree = max(d for _, d in self.graph.degree()) if self.graph.number_of_nodes() > 0 else 1
        return degree / max_degree if max_degree > 0 else 0.0
