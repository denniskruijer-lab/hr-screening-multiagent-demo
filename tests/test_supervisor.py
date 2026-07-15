"""Behavioural eval for the supervisor: dispatch to two worker agents, then
reconcile their scores. No network calls -- every client (supervisor,
screening, calibration) is a scripted fake.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.providers.gemini_adapter import GeminiMessagesClient  # noqa: E402
from src.agents.screening_agent import build_screening_agent  # noqa: E402
from src.agents.supervisor import _make_run_worker, build_supervisor  # noqa: E402
from src.tools import reports  # noqa: E402
from .fakes import (  # noqa: E402
    FakeClient, FakeGeminiSDK, text_response, tool_response,
    gemini_text_response, gemini_tool_response,
)


@pytest.fixture(autouse=True)
def isolate_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "OUTPUT_DIR", tmp_path)


CANDIDATE_ARGS = {
    "candidate_file": "kandidaat_cv.txt",
    "vacancy_file": "vacature.txt",
    "candidate_name": "Jamie Visser",
}


def _screening_report_args(fit_score, estimated_level):
    return {
        "candidate_name": "Jamie Visser",
        "fit_score": fit_score,
        "estimated_level": estimated_level,
        "strengths": ["Python", "RAG side-projects"],
        "gaps": ["No production agent experience"],
        "advice": "Recommend a technical interview focused on agent reliability.",
    }


def test_supervisor_dispatches_both_agents_and_flags_disagreement():
    screening_client = FakeClient([
        tool_response("read_document", {"filename": "kandidaat_cv.txt"}),
        tool_response("get_talent_matrix", {}),
        tool_response("save_screening_report", _screening_report_args(80, "medior")),
        text_response("Screening complete: fit score 80, medior level."),
    ])
    calibration_sdk = FakeGeminiSDK([
        gemini_tool_response("read_document", {"filename": "kandidaat_cv.txt"}),
        gemini_tool_response("get_talent_matrix", {}),
        gemini_tool_response("save_screening_report", _screening_report_args(55, "junior")),
        gemini_text_response("Calibration complete: fit score 55, junior level."),
    ])
    supervisor_sdk = FakeGeminiSDK([
        gemini_tool_response("run_screening_agent", CANDIDATE_ARGS, call_id="call_1"),
        gemini_tool_response("run_calibration_agent", CANDIDATE_ARGS, call_id="call_2"),
        gemini_tool_response(
            "check_score_agreement", {"screening_score": 80, "calibration_score": 55}, call_id="call_3"
        ),
        gemini_text_response("Disagreement detected between the two agents; recommend human review."),
    ])

    supervisor = build_supervisor(
        supervisor_client=GeminiMessagesClient(supervisor_sdk),
        screening_client=screening_client,
        calibration_client=GeminiMessagesClient(calibration_sdk),
    )
    result = supervisor.run("Screen Jamie Visser.")

    assert "human review" in result.final_text
    tool_calls = [step["tool"] for step in result.trace]
    assert tool_calls == ["run_screening_agent", "run_calibration_agent", "check_score_agreement"]
    assert all(step["is_error"] is False for step in result.trace)


def test_run_worker_reports_the_score_it_actually_saved_not_a_guess():
    """The full dispatch test above scripts the supervisor's own tool-call
    arguments directly, so it can't prove the score wasn't just made up by
    the (fake) model. This test checks the real invariant: the observation
    _make_run_worker hands back to the supervisor contains the fit_score
    the worker agent actually persisted via save_screening_report."""
    screening_client = FakeClient([
        tool_response("read_document", {"filename": "kandidaat_cv.txt"}),
        tool_response("get_talent_matrix", {}),
        tool_response("save_screening_report", _screening_report_args(80, "medior")),
        text_response("Screening complete: fit score 80, medior level."),
    ])
    screening_agent = build_screening_agent(screening_client)
    run_worker = _make_run_worker(screening_agent, source="screening")

    observation = run_worker(**CANDIDATE_ARGS)

    assert "[recorded fit_score: 80]" in observation
