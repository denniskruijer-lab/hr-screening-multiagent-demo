"""
Behavioural evals for the agent loop, run without any network calls.

Each test targets one documented failure mode of tool-using agents:
  1. happy path        — loop terminates with a final answer
  2. looping           — runaway tool calling hits the max_iterations guard
  3. incorrect tooluse — unknown tool name is fed back as a correctable error
  4. bad arguments     — missing required args are rejected before execution
  5. crashing tool     — handler exceptions become observations, not crashes
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import AgentResult, MaxIterationsExceeded  # noqa: E402
from .fakes import (  # noqa: E402
    FakeClient, text_response, tool_response, echo_tool, crashing_tool, make_agent,
)


# ── the evals ────────────────────────────────────────────────────────────────
def test_happy_path_terminates_with_final_answer():
    log = []
    client = FakeClient([
        tool_response("echo", {"message": "hi"}),
        text_response("Done: hi was echoed."),
    ])
    result = make_agent(client, [echo_tool(log)]).run("please echo hi")

    assert isinstance(result, AgentResult)
    assert result.final_text == "Done: hi was echoed."
    assert result.iterations == 2
    assert log == ["hi"]
    assert result.trace[0]["is_error"] is False


def test_runaway_tool_loop_hits_max_iterations_guard():
    client = FakeClient([tool_response("echo", {"message": "again"})])  # loops forever
    with pytest.raises(MaxIterationsExceeded):
        make_agent(client, [echo_tool()], max_iterations=3).run("loop please")
    assert len(client.calls) == 3  # guard stopped it exactly at the ceiling


def test_unknown_tool_is_fed_back_as_correctable_error():
    client = FakeClient([
        tool_response("delete_production_db", {"table": "users"}),
        text_response("Understood, that tool does not exist."),
    ])
    result = make_agent(client, [echo_tool()]).run("try a nonexistent tool")

    assert result.trace[0]["is_error"] is True
    # the error observation was sent back to the model as a tool_result
    followup = client.calls[1]["messages"][-1]["content"][0]
    assert followup["is_error"] is True
    assert "Unknown tool" in followup["content"]


def test_missing_required_argument_is_rejected_before_execution():
    log = []
    client = FakeClient([
        tool_response("echo", {}),                       # forgot 'message'
        text_response("I will retry with the argument."),
    ])
    result = make_agent(client, [echo_tool(log)]).run("echo without args")

    assert result.trace[0]["is_error"] is True
    assert log == []  # handler never ran with invalid input


def test_crashing_tool_becomes_observation_not_crash():
    client = FakeClient([
        tool_response("crash", {}),
        text_response("The tool failed, reporting gracefully."),
    ])
    result = make_agent(client, [crashing_tool()]).run("run the crashing tool")

    assert result.trace[0]["is_error"] is True
    assert result.final_text.startswith("The tool failed")
