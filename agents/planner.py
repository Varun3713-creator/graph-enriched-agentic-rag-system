"""
agents/planner.py — LLM-powered Planner Agent
"""
from __future__ import annotations
import logging
from core.llm_client import chat_completion_json
from core.models import AgentAction, ActionType

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """You are an intelligent planning agent for a RAG-based QA system.
Given a user query, decide the best action strategy. 

Return a JSON object with:
{
  "action_type": "RAG_ONLY" | "TOOL_ONLY" | "RAG_AND_TOOL",
  "tool_name": null | "device_status" | "error_lookup",
  "tool_args": {},
  "reasoning": "<brief reasoning>"
}

Guidelines:
- Use RAG_ONLY for: questions about manuals, error codes, documentation, how-to
- Use TOOL_ONLY for: live device status, real-time data requests  
- Use RAG_AND_TOOL for: questions combining manual info + live data
- Use error_lookup tool when query mentions specific error codes like E01, E05 etc.
- Use device_status tool when user asks about current device state
"""


def plan(query: str) -> AgentAction:
    """Use LLM to plan the best action for a query."""
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"User Query: {query}"},
    ]
    try:
        result = chat_completion_json(messages)
        action = AgentAction(
            action_type=ActionType(result.get("action_type", "RAG_ONLY")),
            tool_name=result.get("tool_name"),
            tool_args=result.get("tool_args", {}),
            reasoning=result.get("reasoning", ""),
        )
        logger.info(f"Planner decision: {action.action_type} | tool={action.tool_name}")
        return action
    except Exception as e:
        logger.warning(f"Planner failed: {e}. Defaulting to RAG_ONLY.")
        return AgentAction(action_type=ActionType.RAG_ONLY, reasoning="Fallback to RAG.")
