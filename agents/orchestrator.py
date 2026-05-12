"""
agents/orchestrator.py — Main pipeline orchestrator
Coordinates: Planner → Tools → Retrieval → LLM → Reflection
"""
from __future__ import annotations
import logging
from typing import Generator, List

from core.models import AgentStep, FinalAnswer, ActionType
from core.vector_store import VectorStore
from core.llm_client import chat_completion, embed_texts
from rag.graph import ChunkGraph
from rag.retriever import Retriever
from agents.planner import plan
from agents.tool_agent import execute_tool
from agents.reflection import reflect
from config.settings import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

ANSWER_SYSTEM = """You are a helpful, precise assistant. Answer the user's question using ONLY the provided context.
Be concise but complete. If the context doesn't have enough information, say so clearly.
Always cite the section/source of your information."""


class Orchestrator:
    def __init__(self, vector_store: VectorStore, graph: ChunkGraph):
        self.retriever = Retriever(vector_store, graph)
        self.steps: List[AgentStep] = []

    def _log(self, icon: str, label: str, detail: str, callback: Optional[Callable[[AgentStep], None]] = None) -> AgentStep:
        step = AgentStep(icon=icon, label=label, detail=detail)
        self.steps.append(step)
        logger.info(f"{icon} {label}: {detail}")
        if callback:
            callback(step)
        return step

    def run(self, query: str, step_callback: Optional[Callable[[AgentStep], None]] = None) -> FinalAnswer:
        """
        Run the full agentic pipeline for a query.
        Returns FinalAnswer with answer, confidence, sources, and reasoning steps.
        """
        self.steps = []
        tool_context = ""
        
        def _local_log(i, l, d):
            return self._log(i, l, d, callback=step_callback)

        # ── Step 1: Planner ────────────────────────────────────────────────────
        _local_log("🧠", "Planner", "Analyzing query and deciding action strategy...")
        action = plan(query)
        _local_log(
            "🧠", "Planner Decision",
            f"Action: {action.action_type.value} | {action.reasoning}"
        )

        # ── Step 2: Tool Execution (if needed) ────────────────────────────────
        if action.action_type in (ActionType.TOOL_ONLY, ActionType.RAG_AND_TOOL):
            tool_name = action.tool_name or "device_status"
            _local_log("🔧", "Tool Agent", f"Executing tool: {tool_name} with args {action.tool_args}")
            result = execute_tool(tool_name, action.tool_args)
            if result:
                tool_context = f"\n\n[Tool Result from {tool_name}]:\n{result}"
                _local_log("🔧", "Tool Result", result[:200])
            else:
                _local_log("🔧", "Tool Result", "Tool returned no data.")

        # ── Step 3: Retrieval (if needed) ─────────────────────────────────────
        retrieval_results = []
        context_chunks = []

        if action.action_type in (ActionType.RAG_ONLY, ActionType.RAG_AND_TOOL):
            _local_log("🔍", "Retriever", f"Searching vector index for top-{TOP_K_RETRIEVAL} chunks...")
            retrieval_results = self.retriever.retrieve(query, top_k=TOP_K_RETRIEVAL)
            context_chunks = [r.chunk for r in retrieval_results]

            _local_log(
                "🔍", "Retrieval Complete",
                f"Found {len(retrieval_results)} chunks | "
                f"Top score: {retrieval_results[0].final_score:.3f}" if retrieval_results else "No chunks found"
            )

            expanded = [r for r in retrieval_results if r.similarity_score == 0.0]
            if expanded:
                _local_log("🔗", "Graph Expansion", f"Added {len(expanded)} context chunks via graph links")

        # ── Step 4: Build context and generate answer ─────────────────────────
        context_text = "\n\n---\n\n".join(
            [f"[{c.section} | {c.source}]\n{c.text}" for c in context_chunks[:6]]
        )
        if tool_context:
            context_text += tool_context

        _local_log("🧠", "LLM", "Generating answer from retrieved context...")
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
        ]
        answer = chat_completion(messages, temperature=0.1, max_tokens=1024)

        # ── Step 5: Reflection ────────────────────────────────────────────────
        _local_log("🔁", "Reflection Agent", "Validating answer grounding and completeness...")
        reflection = reflect(query, answer, context_chunks)
        _local_log(
            "📊", "Confidence Score",
            f"{reflection.confidence:.0%} | Grounded: {reflection.is_grounded} | {reflection.feedback}"
        )

        return FinalAnswer(
            answer=answer,
            confidence=reflection.confidence,
            sources=context_chunks,
            steps=self.steps,
            reflection=reflection,
        )
