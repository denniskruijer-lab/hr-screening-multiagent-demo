"""Behavioural evals for the Gemini adapter, run without any network calls.

The Google GenAI SDK's real types (Content, Part, FunctionCall, ...) are used
throughout -- only the network boundary (FakeGeminiSDK.generate_content) is
faked. This means the tests genuinely exercise the translation logic, not a
mocked-away version of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.providers.gemini_adapter import GeminiMessagesClient  # noqa: E402
from .fakes import Block, FakeGeminiSDK, gemini_text_response, gemini_tool_response, echo_tool, make_agent  # noqa: E402


def test_tool_call_response_translates_to_tool_use_stop_reason():
    sdk = FakeGeminiSDK([gemini_tool_response("echo", {"message": "hi"}, call_id="call_1")])
    client = GeminiMessagesClient(sdk)

    response = client.create(model="gemini-test", max_tokens=100, system="test",
                              tools=[echo_tool().spec()], messages=[{"role": "user", "content": "echo hi"}])

    assert response.stop_reason == "tool_use"
    assert len(response.content) == 1
    block = response.content[0]
    assert block.type == "tool_use"
    assert block.name == "echo"
    assert block.input == {"message": "hi"}
    assert block.id == "call_1"


def test_text_only_response_translates_to_end_turn():
    sdk = FakeGeminiSDK([gemini_text_response("Done: hi was echoed.")])
    client = GeminiMessagesClient(sdk)

    response = client.create(model="gemini-test", max_tokens=100, system="test",
                              tools=[], messages=[{"role": "user", "content": "echo hi"}])

    assert response.stop_reason == "end_turn"
    assert response.content[0].type == "text"
    assert response.content[0].text == "Done: hi was echoed."


def test_tool_result_history_round_trip_recovers_tool_name_by_id():
    """A tool_use block followed by its tool_result must translate back into a
    FunctionResponse addressed by the *name* Gemini expects, recovered via id."""
    sdk = FakeGeminiSDK([gemini_text_response("noted")])
    client = GeminiMessagesClient(sdk)

    history = [
        {"role": "user", "content": "echo hi"},
        {"role": "assistant", "content": [
            Block(type="tool_use", name="echo", input={"message": "hi"}, id="call_1"),
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "echo: hi", "is_error": False},
        ]},
    ]
    client.create(model="gemini-test", max_tokens=100, system="test", tools=[], messages=history)

    sent_contents = sdk.calls[-1]["contents"]
    function_response_part = sent_contents[-1].parts[0]
    assert function_response_part.function_response.name == "echo"
    assert function_response_part.function_response.response == {"output": "echo: hi"}


def test_thought_signature_is_captured_and_echoed_back_on_the_next_turn():
    """Regression test: Gemini's newer 'thinking' models reject a follow-up
    call if a prior function_call part's thought_signature isn't echoed back
    exactly (discovered against the real API, not anticipated in advance --
    see the RuntimeError this used to raise with no explanation attached)."""
    sdk = FakeGeminiSDK([
        gemini_tool_response("echo", {"message": "hi"}, call_id="call_1", thought_signature=b"opaque-token"),
        gemini_text_response("noted"),
    ])
    client = GeminiMessagesClient(sdk)

    result = make_agent(client, [echo_tool()]).run("echo hi")

    assert result.final_text == "noted"
    # the second real call's history must carry the same thought_signature forward
    second_call_contents = sdk.calls[1]["contents"]
    assistant_turn = second_call_contents[1]  # [user prompt, assistant tool_use, user tool_result]
    assert assistant_turn.parts[0].thought_signature == b"opaque-token"


def test_agent_runs_a_full_happy_path_through_the_unmodified_agent_class():
    """The money test: Agent (src/agent.py) is never forked for Gemini -- it's
    the exact same class, just handed a GeminiMessagesClient instead."""
    log = []
    sdk = FakeGeminiSDK([
        gemini_tool_response("echo", {"message": "hi"}, call_id="call_1"),
        gemini_text_response("Done: hi was echoed."),
    ])
    client = GeminiMessagesClient(sdk)

    result = make_agent(client, [echo_tool(log)]).run("please echo hi")

    assert result.final_text == "Done: hi was echoed."
    assert log == ["hi"]
    assert result.trace[0]["is_error"] is False
