"""
core/llm_client.py — Azure OpenAI wrapper with retry/backoff
"""
from __future__ import annotations
import time
import json
import logging
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI
from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
)

logger = logging.getLogger(__name__)

_client: Optional[AzureOpenAI] = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    return _client


def chat_completion(
    messages: List[Dict[str, str]],
    deployment: str = AZURE_OPENAI_DEPLOYMENT,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    retries: int = 3,
    response_format: Optional[Dict] = None,
) -> str:
    """Call Azure OpenAI chat completion with exponential backoff."""
    client = _get_client()
    kwargs: Dict[str, Any] = dict(
        model=deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format:
        kwargs["response_format"] = response_format

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning(f"LLM call failed (attempt {attempt+1}): {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("All LLM retry attempts failed.")


def chat_completion_json(
    messages: List[Dict[str, str]],
    deployment: str = AZURE_OPENAI_DEPLOYMENT,
    temperature: float = 0.0,
) -> Any:
    """Return parsed JSON from LLM response."""
    raw = chat_completion(
        messages,
        deployment=deployment,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(raw)


def embed_texts(
    texts: List[str],
    deployment: str = AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    retries: int = 3,
) -> List[List[float]]:
    """Embed a batch of texts, returning a list of float vectors."""
    client = _get_client()
    # Azure has a token/item limit per request — batch in chunks of 16
    BATCH = 16
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        for attempt in range(retries):
            try:
                resp = client.embeddings.create(model=deployment, input=batch)
                all_embeddings.extend([d.embedding for d in resp.data])
                break
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(f"Embedding failed (attempt {attempt+1}): {exc}. Retrying in {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Embedding batch {i//BATCH} failed after retries.")

    return all_embeddings
