"""A small in-memory vector store: chunk, embed once, cosine-search.

No vector database needed at this scale (a handful of documents) -- a plain
list and a hand-rolled cosine similarity are enough, and far easier to read
and explain than pulling in a dependency for it.
"""
from __future__ import annotations

import logging
import math

from ..tools.documents import read_document
from .embeddings import embed_texts

logger = logging.getLogger("hr_agent")

CHUNK_SIZE = 300  # characters, not tokens -- good enough at this scale


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks of roughly chunk_size characters, on word boundaries."""
    words = text.split()
    chunks = []
    current: list[str] = []
    current_length = 0
    for word in words:
        current.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_corpus(filenames: list[str]) -> list[str]:
    """Read each data/ filename and chunk it, for the vector store to embed."""
    chunks: list[str] = []
    for filename in filenames:
        chunks.extend(chunk_text(read_document(filename)))
    return chunks


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def _keyword_overlap_score(query: str, chunk: str) -> int:
    query_words = set(query.lower().split())
    chunk_words = set(chunk.lower().split())
    return len(query_words & chunk_words)


class InMemoryVectorStore:
    """Embeds a fixed set of chunks once at construction time, then answers
    nearest-neighbour queries against them.

    Falls back to naive keyword overlap if embeddings are unavailable, either
    at construction time or for a specific query later -- see embeddings.py.
    """

    def __init__(self, chunks: list[str], client):
        self._chunks = chunks
        self._client = client
        self._vectors = embed_texts(chunks, client)  # None if the embedding call failed
        if self._vectors is None:
            logger.warning("Embeddings unavailable at startup -- retrieval will use keyword overlap instead.")

    def search(self, query: str, k: int = 3) -> list[str]:
        if not self._chunks:
            return []

        if self._vectors is not None:
            query_vectors = embed_texts([query], self._client)
            if query_vectors is not None:
                return self._search_by_similarity(query_vectors[0], k)

        return self._search_by_keyword_overlap(query, k)

    def _search_by_similarity(self, query_vector: list[float], k: int) -> list[str]:
        scored = [
            (_cosine_similarity(query_vector, vector), chunk)
            for vector, chunk in zip(self._vectors, self._chunks)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]

    def _search_by_keyword_overlap(self, query: str, k: int) -> list[str]:
        scored = [(_keyword_overlap_score(query, chunk), chunk) for chunk in self._chunks]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]
