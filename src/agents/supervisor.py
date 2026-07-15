"""The supervisor: dispatches to the screening and calibration agents, then
reconciles their scores.

This is the multi-agent seam. The supervisor is itself a plain Agent (see
src/agent.py) -- its "tools" happen to be other agents' run() methods, wrapped
in closures. No new orchestration primitive was needed: agents calling agents
through the same Tool abstraction that already existed for calling functions.

Other multi-agent patterns (a fully decentralized network of peer agents, or a
hierarchy of supervisors) were considered and are documented, not built --
see docs/architecture-alternatives.md.
"""
from __future__ import annotations

from ..agent import Agent, Tool
from ..tools.reports import flag_disagreement, load_report
from ..tools.retrieval import build_retrieve_context_tool
from .calibration_agent import build_calibration_agent
from .screening_agent import build_screening_agent

MODEL = "gemini-flash-lite-latest"  # cheapest/fastest tier: this agent only dispatches and aggregates
MAX_TOKENS = 8000  # same safety margin as the worker agents -- see screening_agent.py's comment

SYSTEM_PROMPT = """You are the supervisor of a two-agent candidate screening pipeline.

Given a candidate CV and a vacancy, do the following in order:
1. Call run_screening_agent to get the primary screening assessment.
2. Call run_calibration_agent to get an independent second opinion.
3. Call check_score_agreement with both agents' recorded fit scores.
4. Summarise both assessments and the agreement check for the user.

You do not assess the candidate yourself -- that is the worker agents' job.
Your job is dispatch, aggregation, and reconciliation."""


def _make_run_worker(worker_agent: Agent, source: str):
    """Wrap a worker agent's run() as a tool handler.

    After the worker agent finishes and saves its report, the numeric fit_score
    is read straight back from the saved file -- rather than asking the
    supervisor model to copy a number out of a paragraph of prose -- so the
    score `check_score_agreement` receives later is exactly what was saved,
    not a value the model might mis-transcribe.
    """
    def run_worker(candidate_file: str, vacancy_file: str, candidate_name: str) -> str:
        result = worker_agent.run(
            f"Screen the candidate in '{candidate_file}' against the vacancy in "
            f"'{vacancy_file}'. Use the talent matrix for level calibration and save a report."
        )
        report = load_report(candidate_name, source=source)
        return f"{result.final_text}\n\n[recorded fit_score: {report['fit_score']}]"
    return run_worker


def _check_score_agreement(screening_score: int, calibration_score: int) -> str:
    return flag_disagreement(screening_score, calibration_score)


def build_supervisor(supervisor_client, screening_client, calibration_client, vector_store=None) -> Agent:
    """All three clients are injected independently, so each can be faked in tests.

    `supervisor_client` drives the supervisor's own dispatch/aggregation calls;
    `screening_client` and `calibration_client` are handed to the two worker
    agents this function builds internally. `vector_store` is optional: if
    given, both worker agents also get a retrieve_context tool backed by it
    (RAG); if omitted, they work exactly as before -- so existing tests that
    don't care about retrieval don't need to change.
    """
    retrieval_tools = [build_retrieve_context_tool(vector_store)] if vector_store is not None else []
    screening_agent = build_screening_agent(screening_client, extra_tools=retrieval_tools)
    calibration_agent = build_calibration_agent(calibration_client, extra_tools=retrieval_tools)

    candidate_task_schema = {
        "type": "object",
        "properties": {
            "candidate_file": {"type": "string", "description": "CV filename inside data/"},
            "vacancy_file": {"type": "string", "description": "Vacancy filename inside data/"},
            "candidate_name": {"type": "string", "description": "Candidate's name, used to find the saved report"},
        },
        "required": ["candidate_file", "vacancy_file", "candidate_name"],
    }

    tools = [
        Tool(
            name="run_screening_agent",
            description="Run the primary screening agent (Claude) on a candidate.",
            input_schema=candidate_task_schema,
            handler=_make_run_worker(screening_agent, source="screening"),
        ),
        Tool(
            name="run_calibration_agent",
            description="Run the independent calibration agent (Gemini) on the same candidate.",
            input_schema=candidate_task_schema,
            handler=_make_run_worker(calibration_agent, source="calibration"),
        ),
        Tool(
            name="check_score_agreement",
            description="Compare the screening and calibration fit scores and flag if they disagree.",
            input_schema={
                "type": "object",
                "properties": {
                    "screening_score": {"type": "integer"},
                    "calibration_score": {"type": "integer"},
                },
                "required": ["screening_score", "calibration_score"],
            },
            handler=_check_score_agreement,
        ),
    ]
    return Agent(client=supervisor_client, model=MODEL, tools=tools, system_prompt=SYSTEM_PROMPT, max_tokens=MAX_TOKENS)
