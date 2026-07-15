"""The calibration agent: an independent second opinion on the same candidate.

Runs on Gemini, deliberately a different provider than the screening agent --
the point of a second opinion is that it should not share the first model's
blind spots, which a second Claude call would not guarantee.
"""
from __future__ import annotations

from ..agent import Agent, Tool
from ..tools import reports
from .common_tools import SAVE_REPORT_SCHEMA, build_document_reading_tools

MODEL = "gemini-2.5-pro"

SYSTEM_PROMPT = """You are an independent calibration agent for a recruitment screening pipeline.

Another agent has already screened this candidate, but you must NOT assume
their conclusion -- form your own judgement from the source documents.

Work step by step:
1. Read the vacancy and the candidate CV with read_document.
2. Fetch the talent matrix with get_talent_matrix.
3. Independently assess fit against the vacancy and the matrix.
4. Save your own conclusion with save_screening_report, then summarise it briefly.

Ground every claim in the documents you actually read."""


def _save_report(candidate_name, fit_score, estimated_level, strengths, gaps, advice) -> str:
    return reports.save_screening_report(
        candidate_name, fit_score, estimated_level, strengths, gaps, advice, source="calibration"
    )


def build_calibration_agent(client) -> Agent:
    """`client` is injected: a real Gemini-backed client in production, a fake in tests."""
    tools = build_document_reading_tools() + [
        Tool(
            name="save_screening_report",
            description="Save your independent structured screening report. All fields are validated.",
            input_schema=SAVE_REPORT_SCHEMA,
            handler=_save_report,
        ),
    ]
    return Agent(client=client, model=MODEL, tools=tools, system_prompt=SYSTEM_PROMPT)
