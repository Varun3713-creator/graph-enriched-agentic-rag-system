"""
rag/chunking.py — Adaptive chunking engine
"""
from __future__ import annotations
import re
import logging
from typing import List, Dict, Any, Optional

from core.models import Chunk, ChunkType
from rag.table_extractor import extract_tables_from_text, table_to_chunk_text
from config.settings import MAX_CHUNK_TOKENS, CHUNK_OVERLAP, SLIDING_WINDOW_SIZE

logger = logging.getLogger(__name__)


# ── Token estimation (without tiktoken dependency on every import) ─────────────

def _token_count(text: str) -> int:
    """Rough token count: words * 1.3"""
    return int(len(text.split()) * 1.3)


# ── Heading detection ──────────────────────────────────────────────────────────

HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+(.+)"),                    # Markdown headings
    re.compile(r"^(\d+[\.\)]\s+[A-Z].{3,60})$"),      # Numbered headings
    re.compile(r"^([A-Z][A-Z\s\-]{4,50})$"),           # ALL-CAPS headings
    re.compile(r"^([A-Z][^.!?]{5,60}):?\s*$"),         # Title-case short lines
]


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return any(p.match(stripped) for p in HEADING_PATTERNS)


def _extract_heading_text(line: str) -> str:
    stripped = line.strip()
    for p in HEADING_PATTERNS:
        m = p.match(stripped)
        if m:
            return m.group(1).strip() if m.lastindex else stripped
    return stripped


# ── List detection ────────────────────────────────────────────────────────────

LIST_PATTERN = re.compile(r"^(\s*[\-\*\•]\s+|\s*\d+[\.\)]\s+)")

def _is_list_item(line: str) -> bool:
    return bool(LIST_PATTERN.match(line))


# ── Main chunker ──────────────────────────────────────────────────────────────

class AdaptiveChunker:
    def __init__(self, source: str = ""):
        self.source = source

    def chunk(self, text: str, structure: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Main entry point: produce chunks from raw text.
        Uses structure hints if provided.
        """
        chunks: List[Chunk] = []
        lines = text.split("\n")

        # Extract tables first (they get separate chunks)
        tables = extract_tables_from_text(text)
        table_raw_set = set()
        for tbl in tables:
            table_raw_set.update(tbl["raw_text"].split("\n"))

        current_section = "General"
        current_subsection = ""
        paragraph_buffer: List[str] = []
        list_buffer: List[str] = []
        chunk_idx = 0

        def flush_paragraph():
            nonlocal chunk_idx
            text_out = " ".join(paragraph_buffer).strip()
            if not text_out:
                return
            for sub_chunk in _sliding_window(text_out, chunk_idx, current_section, current_subsection, self.source):
                chunks.append(sub_chunk)
                chunk_idx += 1
            paragraph_buffer.clear()

        def flush_list():
            nonlocal chunk_idx
            if not list_buffer:
                return
            text_out = "\n".join(list_buffer).strip()
            c = Chunk(
                id=f"{self.source}_{chunk_idx}",
                text=text_out,
                section=current_section,
                subsection=current_subsection,
                chunk_type=ChunkType.LIST,
                source=self.source,
                position=chunk_idx,
            )
            chunks.append(c)
            chunk_idx += 1
            list_buffer.clear()

        for line in lines:
            raw = line.rstrip()
            stripped = raw.strip()

            # Skip blank lines — flush pending buffers
            if not stripped:
                if list_buffer:
                    flush_list()
                elif paragraph_buffer:
                    flush_paragraph()
                continue

            # Skip table lines (handled separately)
            if stripped in table_raw_set:
                continue

            # Heading
            if _is_heading(raw):
                flush_list()
                flush_paragraph()
                current_section = _extract_heading_text(raw)
                current_subsection = ""
                # Create anchor chunk for the heading itself
                c = Chunk(
                    id=f"{self.source}_{chunk_idx}",
                    text=current_section,
                    section=current_section,
                    chunk_type=ChunkType.HEADING,
                    source=self.source,
                    position=chunk_idx,
                )
                chunks.append(c)
                chunk_idx += 1
                continue

            # List item
            if _is_list_item(raw):
                if paragraph_buffer:
                    flush_paragraph()
                list_buffer.append(stripped)
                continue

            # Regular paragraph text
            if list_buffer:
                flush_list()
            paragraph_buffer.append(stripped)

        # Flush remaining
        flush_list()
        flush_paragraph()

        # Add table chunks
        for tbl in tables:
            text_out = table_to_chunk_text(tbl)
            c = Chunk(
                id=f"{self.source}_{chunk_idx}",
                text=text_out,
                section=current_section,
                chunk_type=ChunkType.TABLE,
                source=self.source,
                position=chunk_idx,
                metadata={"json_data": tbl["json_data"]},
            )
            chunks.append(c)
            chunk_idx += 1

        logger.info(f"Chunked '{self.source}': {len(chunks)} chunks produced")
        return chunks


def _sliding_window(
    text: str,
    start_idx: int,
    section: str,
    subsection: str,
    source: str,
) -> List[Chunk]:
    """Split long text using a sliding window approach."""
    if _token_count(text) <= MAX_CHUNK_TOKENS:
        return [
            Chunk(
                id=f"{source}_{start_idx}",
                text=text,
                section=section,
                subsection=subsection,
                chunk_type=ChunkType.SEMANTIC,
                source=source,
                position=start_idx,
            )
        ]

    words = text.split()
    chunks = []
    i = 0
    idx_offset = 0

    while i < len(words):
        window = words[i : i + SLIDING_WINDOW_SIZE]
        chunk_text = " ".join(window)
        chunks.append(
            Chunk(
                id=f"{source}_{start_idx + idx_offset}",
                text=chunk_text,
                section=section,
                subsection=subsection,
                chunk_type=ChunkType.SEMANTIC,
                source=source,
                position=start_idx + idx_offset,
            )
        )
        idx_offset += 1
        i += SLIDING_WINDOW_SIZE - CHUNK_OVERLAP

    return chunks
