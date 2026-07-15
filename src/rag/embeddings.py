"""Embeddings via the Gemini API.

Failures degrade gracefully (return None) rather than raise -- retrieval
becomes unavailable, it does not crash the run. This mirrors agent.py's
"crashing tools become observations, never crashes" philosophy one layer up.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("hr_agent")

EMBEDDING_MODEL = "gemini-embedding-001"


def embed_texts(texts: list[str], client) -> list[list[float]] | None:
    """Embed a batch of texts. Returns None (never raises) if the call fails."""
    if not texts:
        return []
    try:
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
        return [embedding.values for embedding in response.embeddings]
    except Exception as exc:
        logger.error("Gemini embedding call failed, retrieval will degrade: %s", exc)
        return None
