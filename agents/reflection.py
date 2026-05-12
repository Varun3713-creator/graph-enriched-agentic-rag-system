"""
agents/reflection.py — Reflection Agent for answer validation and confidence scoring
"""
from __future__ import annotations
import logging
from typing import List

from core.llm_client import chat_completion_json
from core.models import Chunk, ReflectionResult

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM = """You are a critical reflection agent for a RAG system.
Your job is to evaluate whether an answer is grounded in the provided context and complete.

Return a JSON object:
{
  "is_grounded": true | false,
  "is_complete": true | false,
  "confidence": 0.0 to 1.0,
  "feedback": "<brief feedback>"
}

Scoring guidance:
- confidence 0.9-1.0: Answer fully supported by context, no gaps
- confidence 0.7-0.9: Mostly supported, minor gaps
- confidence 0.5-0.7: Partially supported, some speculation
- confidence 0.0-0.5: Poorly grounded or incomplete
"""


def reflect(
    query: str,
    answer: str,
    context_chunks: List[Chunk],
) -> ReflectionResult:
    """
    Validate an answer against the retrieved context.
    Returns ReflectionResult with confidence score and feedback.
    """
    context_text = "\n\n---\n\n".join(
        [f"[{c.section}] {c.text[:400]}" for c in context_chunks[:5]]
    )

    messages = [
        {"role": "system", "content": REFLECTION_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User Query: {query}\n\n"
                f"Retrieved Context:\n{context_text}\n\n"
                f"Generated Answer:\n{answer}"
            ),
        },
    ]

    try:
        result = chat_completion_json(messages)
        return ReflectionResult(
            is_grounded=bool(result.get("is_grounded", True)),
            is_complete=bool(result.get("is_complete", True)),
            confidence=float(result.get("confidence", 0.75)),
            feedback=str(result.get("feedback", "")),
        )
    except Exception as e:
        logger.warning(f"Reflection failed: {e}. Using default.")
        return ReflectionResult(
            is_grounded=True,
            is_complete=True,
            confidence=0.7,
            feedback="Reflection unavailable.",
        )
