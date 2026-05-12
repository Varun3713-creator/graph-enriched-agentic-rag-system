"""
agents/tool_agent.py — Tool execution agent
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from tools.device_api import get_device_status
from tools.error_lookup import lookup_error

logger = logging.getLogger(__name__)

TOOL_REGISTRY: Dict[str, Any] = {
    "device_status": get_device_status,
    "error_lookup": lookup_error,
}


def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
    """
    Execute a registered tool and return result as a string.
    Returns None if tool not found or fails.
    """
    if tool_name not in TOOL_REGISTRY:
        logger.warning(f"Tool '{tool_name}' not found in registry.")
        return None

    fn = TOOL_REGISTRY[tool_name]
    try:
        result = fn(**tool_args)
        logger.info(f"Tool '{tool_name}' executed successfully.")
        return str(result)
    except Exception as e:
        logger.error(f"Tool '{tool_name}' execution failed: {e}")
        return f"Tool execution error: {e}"
