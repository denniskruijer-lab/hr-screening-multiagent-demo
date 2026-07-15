"""The screening agent: the primary CV-vs-vacancy assessment. Runs on Claude."""
from __future__ import annotations

from ..agent import Agent, Tool
from ..tools import reports
from .common_tools import SAVE_REPORT_SCHEMA, build_document_reading_tools

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are a recruitment screening agent for a technology consultancy.

Your job: assess how well a candidate fits a vacancy, calibrated against the
company's talent matrix.

Work step by step:
1. Read the vacancy and the candidate CV with read_document.
2. Fetch the talent matrix with get_talent_matrix.
3. Compare the candidate against both the vacancy requirements and the matrix.
4. Save your conclusion with save_screening_report, then summarise it briefly.

Ground every claim in the documents you actually read. If information is
missing, say so in the report instead of guessing."""


def _save_report(candidate_name, fit_score, estimated_level, strengths, gaps, advice) -> str:
    # source is fixed to "screening" here, not left for the model to set --
    # which report this is belongs to the code that built this agent, not the prompt.
    return reports.save_screening_report(
        candidate_name, fit_score, estimated_level, strengths, gaps, advice, source="screening"
    )


def build_screening_agent(client, extra_tools: list[Tool] | None = None) -> Agent:
    """`client` is injected: a real Anthropic client in production, a fake in tests.

    `extra_tools` lets the caller add capabilities (e.g. RAG retrieval)
    without this module needing to know anything about how they're built.
    """
    tools = build_document_reading_tools() + [
        Tool(
            name="save_screening_report",
            description="Save the final structured screening report. All fields are validated.",
            input_schema=SAVE_REPORT_SCHEMA,
            handler=_save_report,
        ),
    ] + (extra_tools or [])
    return Agent(client=client, model=MODEL, tools=tools, system_prompt=SYSTEM_PROMPT)
