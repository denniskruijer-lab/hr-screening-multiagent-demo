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

from src.agent import Agent, AgentResult, MaxIterationsExceeded, Tool  # noqa: E402


# ── a tiny scripted stand-in for the Anthropic client ───────────────────────
class Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def text_response(text):
    return Block(stop_reason="end_turn", content=[Block(type="text", text=text)])


def tool_response(name, tool_input, tool_id="tu_1"):
    return Block(stop_reason="tool_use",
                 content=[Block(type="tool_use", name=name, input=tool_input, id=tool_id)])


class FakeClient:
    """Returns scripted responses in order; repeats the last one when exhausted."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = self  # so client.messages.create(...) works

    def create(self, **kwargs):
        # Snapshot: the agent mutates its messages list in place, so store a
        # copy of the state *at call time*, not a reference that keeps growing.
        snapshot = dict(kwargs)
        snapshot["messages"] = list(kwargs["messages"])
        self.calls.append(snapshot)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


# ── shared fixtures ──────────────────────────────────────────────────────────
def echo_tool(calls_log=None):
    def handler(message: str) -> str:
        if calls_log is not None:
            calls_log.append(message)
        return f"echo: {message}"
    return Tool(
        name="echo",
        description="Echo a message back.",
        input_schema={"type": "object",
                      "properties": {"message": {"type": "string"}},
                      "required": ["message"]},
        handler=handler,
    )


def crashing_tool():
    def handler() -> str:
        raise RuntimeError("database on fire")
    return Tool(name="crash", description="Always fails.",
                input_schema={"type": "object", "properties": {}}, handler=handler)


def make_agent(client, tools, max_iterations=5):
    return Agent(client=client, model="test-model", tools=tools,
                 system_prompt="test", max_iterations=max_iterations)


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
