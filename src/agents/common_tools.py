"""Tool definitions shared by both worker agents (screening and calibration).

Both agents read the same kind of documents the same way -- only what they do
with the content differs (that lives in each agent's system prompt), so the
two "read" tools are built once here rather than duplicated per agent.
"""
from __future__ import annotations

from ..agent import Tool
from ..tools import documents


def build_document_reading_tools() -> list[Tool]:
    return [
        Tool(
            name="read_document",
            description="Read a text document (vacancy or CV) from the data directory.",
            input_schema={
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "File name inside data/"}},
                "required": ["filename"],
            },
            handler=documents.read_document,
        ),
        Tool(
            name="get_talent_matrix",
            description="Get the competence matrix (levels x competencies) used for calibration.",
            input_schema={"type": "object", "properties": {}},
            handler=documents.get_talent_matrix,
        ),
    ]


SAVE_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_name": {"type": "string"},
        "fit_score": {"type": "integer", "description": "0-100"},
        "estimated_level": {"type": "string", "enum": ["junior", "medior", "senior"]},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "advice": {"type": "string", "description": "Concrete next step, >= 20 chars"},
    },
    "required": ["candidate_name", "fit_score", "estimated_level", "strengths", "gaps", "advice"],
}
