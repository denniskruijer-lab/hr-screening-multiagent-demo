"""Evals for src/rag/: embeddings and the in-memory vector store.

No network calls -- FakeEmbeddingSDK is deterministic, FailingEmbeddingSDK
exercises the "embeddings unavailable" degradation path.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.embeddings import embed_texts  # noqa: E402
from src.rag.store import InMemoryVectorStore, chunk_text  # noqa: E402
from .fakes import FailingEmbeddingSDK, FakeEmbeddingSDK  # noqa: E402

CHUNKS = [
    "This project uses Python and Docker heavily for the backend services.",
    "We write SQL queries every day for the reporting pipeline.",
    "The frontend is built with TypeScript and a component library.",
]


def test_embed_texts_returns_none_on_failure():
    assert embed_texts(["hello"], FailingEmbeddingSDK()) is None


def test_embed_texts_returns_a_vector_per_text():
    vectors = embed_texts(["python code", "sql query"], FakeEmbeddingSDK())
    assert len(vectors) == 2


def test_search_ranks_by_shared_vocabulary_when_embeddings_available():
    store = InMemoryVectorStore(CHUNKS, FakeEmbeddingSDK())
    results = store.search("python docker experience", k=1)
    assert "Docker" in results[0]


def test_search_falls_back_to_keyword_overlap_when_embeddings_unavailable():
    store = InMemoryVectorStore(CHUNKS, FailingEmbeddingSDK())
    assert store._vectors is None  # embeddings genuinely unavailable, not just untried

    results = store.search("typescript frontend component", k=1)
    assert "TypeScript" in results[0]


def test_chunk_text_recovers_all_words_across_chunks():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=50)
    assert len(chunks) > 1
    assert " ".join(chunks).split() == text.split()
