"""Shared test doubles: no test in this project ever makes a real network call.

FakeClient stands in for the Anthropic client; FakeGeminiSDK stands in for the
google-genai client. Both are scripted (return pre-programmed responses in
order) so agent behaviour is verified deterministically.
"""
from __future__ import annotations

from src.agent import Agent, Tool


class Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def text_response(text):
    return Block(stop_reason="end_turn", content=[Block(type="text", text=text)])


def tool_response(name, tool_input, tool_id="tu_1"):
    return Block(stop_reason="tool_use",
                 content=[Block(type="tool_use", name=name, input=tool_input, id=tool_id)])


class FakeClient:
    """Returns scripted Anthropic-shape responses in order; repeats the last one when exhausted."""
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


# ── fakes for the Gemini side ────────────────────────────────────────────────
class FakePart:
    """A minimal stand-in for google.genai.types.Part, as returned in a response."""
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class FakeFunctionCall:
    def __init__(self, name, args, id=None):
        self.name = name
        self.args = args
        self.id = id


class FakeGeminiResponse:
    """A minimal stand-in for google.genai.types.GenerateContentResponse."""
    def __init__(self, parts):
        content = Block(parts=parts)
        candidate = Block(content=content)
        self.candidates = [candidate]


def gemini_text_response(text):
    return FakeGeminiResponse(parts=[FakePart(text=text)])


def gemini_tool_response(name, args, call_id="call_1"):
    return FakeGeminiResponse(parts=[FakePart(function_call=FakeFunctionCall(name=name, args=args, id=call_id))])


class FakeGeminiSDK:
    """Scripted stand-in for a google-genai Client: exposes .models.generate_content(...)."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.models = self  # so sdk_client.models.generate_content(...) works

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


# ── shared agent fixtures (provider-agnostic -- Agent doesn't care which client it got) ──
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
