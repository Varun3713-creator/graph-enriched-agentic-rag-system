"""
core/models.py — Pydantic data models for the entire system
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# ── Chunk ────────────────────────────────────────────────────────────────────

class ChunkType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    SEMANTIC = "semantic_chunk"


class Chunk(BaseModel):
    id: str
    text: str
    section: str = ""
    subsection: str = ""
    chunk_type: ChunkType = ChunkType.SEMANTIC
    source: str = ""
    page: int = 0
    position: int = 0          # ordinal index within document
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    class Config:
        arbitrary_types_allowed = True


# ── Graph ─────────────────────────────────────────────────────────────────────

class LinkType(str, Enum):
    NEXT = "NEXT"
    PARENT = "PARENT"
    SIMILAR = "SIMILAR"
    ENTITY_MATCH = "ENTITY_MATCH"


class ChunkLink(BaseModel):
    source_id: str
    target_id: str
    link_type: LinkType
    weight: float = 1.0


# ── Retrieval ────────────────────────────────────────────────────────────────

class RetrievalResult(BaseModel):
    chunk: Chunk
    similarity_score: float
    section_score: float = 0.0
    graph_score: float = 0.0
    position_score: float = 0.0
    final_score: float = 0.0


# ── Agent ────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    RAG_ONLY = "RAG_ONLY"
    TOOL_ONLY = "TOOL_ONLY"
    RAG_AND_TOOL = "RAG_AND_TOOL"


class AgentAction(BaseModel):
    action_type: ActionType
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class ReflectionResult(BaseModel):
    is_grounded: bool
    is_complete: bool
    confidence: float          # 0.0 – 1.0
    feedback: str = ""


class AgentStep(BaseModel):
    icon: str
    label: str
    detail: str


class FinalAnswer(BaseModel):
    answer: str
    confidence: float
    sources: List[Chunk]
    steps: List[AgentStep]
    reflection: ReflectionResult
