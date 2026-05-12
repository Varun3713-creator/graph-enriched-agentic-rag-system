"""
rag/structure_detector.py — LLM-assisted document structure detection
"""
from __future__ import annotations
import json
import logging
from typing import Any, Dict

from core.llm_client import chat_completion_json

logger = logging.getLogger(__name__)

STRUCTURE_PROMPT = """You are a document structure analyzer. Given the following document text, extract its structure.

Return a JSON object with this schema:
{{
  "sections": [
    {{
      "title": "<section title>",
      "level": 1,
      "subsections": [
        {{
          "title": "<subsection title>",
          "level": 2,
          "content_types": ["paragraph", "list", "table"],
          "has_table": false,
          "key_entities": ["entity1", "entity2"]
        }}
      ]
    }}
  ],
  "document_type": "<manual|report|specification|other>",
  "main_topics": ["topic1", "topic2"]
}}

Document text (first 3000 chars):
{text}
"""


def detect_structure(text: str) -> Dict[str, Any]:
    """
    Use LLM to detect document structure.
    Returns a structured dict with sections, subsections, content types.
    """
    prompt_text = text[:3000]
    messages = [
        {"role": "system", "content": "You are a precise document structure analyzer. Return only valid JSON."},
        {"role": "user", "content": STRUCTURE_PROMPT.format(text=prompt_text)},
    ]
    try:
        result = chat_completion_json(messages)
        logger.info(f"Structure detected: {len(result.get('sections', []))} sections, type={result.get('document_type')}")
        return result
    except Exception as e:
        logger.warning(f"Structure detection failed: {e}. Using fallback.")
        return _fallback_structure(text)


def _fallback_structure(text: str) -> Dict[str, Any]:
    """Simple regex-based fallback structure detection."""
    import re
    sections = []
    lines = text.split("\n")
    current_section = None

    for line in lines:
        stripped = line.strip()
        # Detect heading-like lines (ALL CAPS, numbered, or short lines ending with ":")
        if stripped and (
            re.match(r"^\d+[\.\)]\s+\w", stripped)
            or re.match(r"^[A-Z][A-Z\s]{4,}$", stripped)
            or (len(stripped) < 60 and stripped.endswith(":"))
        ):
            current_section = {"title": stripped, "level": 1, "subsections": []}
            sections.append(current_section)

    return {
        "sections": sections,
        "document_type": "manual",
        "main_topics": [],
    }
