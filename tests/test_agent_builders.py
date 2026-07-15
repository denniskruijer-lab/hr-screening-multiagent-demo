"""Evals for the agent builder functions (src/agents/*.py) themselves --
not their runtime behaviour (covered elsewhere), just how they configure Agent.

Regression test for a real bug: Agent's default max_tokens=2000 was fine for
the short fictional sample CV, but Claude Sonnet 5's extended thinking can
consume an entire 2000-token budget on a longer, information-dense real CV,
leaving stop_reason="max_tokens" with no text or tool call produced at all
(confirmed against the real API, not just theorised). Each builder must
configure a generous enough max_tokens that this can't silently regress.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.calibration_agent import build_calibration_agent  # noqa: E402
from src.agents.screening_agent import build_screening_agent  # noqa: E402
from src.agents.supervisor import build_supervisor  # noqa: E402
from .fakes import FakeClient  # noqa: E402

MIN_SAFE_MAX_TOKENS = 8000  # the value verified against the real API to fix the bug above


def test_screening_agent_has_a_generous_max_tokens_budget():
    agent = build_screening_agent(FakeClient([]))
    assert agent.max_tokens >= MIN_SAFE_MAX_TOKENS


def test_calibration_agent_has_a_generous_max_tokens_budget():
    agent = build_calibration_agent(FakeClient([]))
    assert agent.max_tokens >= MIN_SAFE_MAX_TOKENS


def test_supervisor_has_a_generous_max_tokens_budget():
    client = FakeClient([])
    supervisor = build_supervisor(supervisor_client=client, screening_client=client, calibration_client=client)
    assert supervisor.max_tokens >= MIN_SAFE_MAX_TOKENS
