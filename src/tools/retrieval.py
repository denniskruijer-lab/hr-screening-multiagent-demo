"""The retrieve_context tool: lets an agent pull relevant chunks out of the
vector store on demand, instead of the whole corpus being stuffed into every
prompt. Keeping this a tool call (rather than pre-injected context) keeps
retrieval visibly agentic and independently testable.
"""
from __future__ import annotations

from ..agent import Tool


def build_retrieve_context_tool(store) -> Tool:
    def retrieve_context(query: str) -> str:
        chunks = store.search(query, k=3)
        if not chunks:
            return "No relevant context found."
        return "\n\n---\n\n".join(chunks)

    return Tool(
        name="retrieve_context",
        description=(
            "Retrieve the most relevant chunks of context (from the vacancy, CV, "
            "and talent matrix) for a specific query, instead of re-reading whole documents."
        ),
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=retrieve_context,
    )
