"""Adapter that lets a Gemini client be used as the `client` argument of Agent.

Agent (see src/agent.py) only ever calls `client.messages.create(model=, max_tokens=,
system=, tools=, messages=)` and expects back an object with a `.stop_reason` and a
`.content` list of blocks (`.type` plus either `.text`, or `.name`/`.input`/`.id`).
That is a duck-typed shape, not an Anthropic class -- so this module translates
between it and the Google GenAI SDK's own shapes, and Agent itself never needs to
know which provider it is talking to.

Design choices worth calling out:
- The full message history is re-translated on every call, the same way Agent
  re-sends its whole `messages` list every turn. There is no state kept between
  calls, which keeps this adapter as easy to reason about (and test) as Agent itself.
- A failed provider call (network error, rate limit, malformed response) is
  re-raised as a plain RuntimeError rather than turned into a tool observation --
  that is a different failure class than the tool-execution guards Agent already
  has (see the "Honest scope" section in README.md).
"""
from __future__ import annotations

import logging
import os

from google.genai import types

logger = logging.getLogger("hr_agent")


def build_gemini_sdk_client():
    """Construct a real google-genai client from GEMINI_API_KEY or GOOGLE_API_KEY."""
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Get a key at "
            "https://aistudio.google.com/apikey and set it as an environment variable "
            "before running against the real API."
        )

    from google import genai  # imported lazily -- keeps this module importable without the SDK installed

    return genai.Client()


# ── duck-typed response blocks, matching what Agent expects from Anthropic ──────
class TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class ToolUseBlock:
    def __init__(self, name: str, tool_input: dict, block_id: str):
        self.type = "tool_use"
        self.name = name
        self.input = tool_input
        self.id = block_id


class GeminiResponse:
    def __init__(self, stop_reason: str, content: list):
        self.stop_reason = stop_reason
        self.content = content


class GeminiMessagesClient:
    """Wraps a google-genai SDK client so it satisfies Agent's `client.messages.create(...)` shape."""

    def __init__(self, sdk_client):
        self._sdk = sdk_client
        self.messages = self  # mirrors the FakeClient trick in tests/fakes.py

    def create(self, *, model: str, max_tokens: int, system: str, tools: list[dict], messages: list[dict]):
        gemini_tools = _translate_tools(tools)
        contents, _ = _translate_history(messages)
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=gemini_tools,
            max_output_tokens=max_tokens,
        )
        try:
            raw = self._sdk.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            logger.error("Gemini call failed: %s", exc)
            raise RuntimeError(f"Gemini call failed: {exc}") from exc
        return _translate_response(raw)


# ── translation helpers ─────────────────────────────────────────────────────────
def _translate_tools(tool_specs: list[dict]) -> list["types.Tool"]:
    """Anthropic-shape tool specs (name/description/input_schema) -> a Gemini Tool."""
    declarations = [
        types.FunctionDeclaration(
            name=spec["name"],
            description=spec["description"],
            parameters_json_schema=spec["input_schema"],
        )
        for spec in tool_specs
    ]
    return [types.Tool(function_declarations=declarations)]


def _translate_history(messages: list[dict]) -> tuple[list["types.Content"], dict[str, str]]:
    """Anthropic-shape message history -> Gemini Content list.

    Also returns `id_to_name`: the tool_use id -> tool name mapping built up
    along the way, needed to translate a later tool_result back into a Gemini
    FunctionResponse (which is addressed by name, not by id).
    """
    contents: list[types.Content] = []
    id_to_name: dict[str, str] = {}

    for message in messages:
        content = message["content"]

        if isinstance(content, str):
            # The very first user message: a plain prompt string.
            contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            continue

        if message["role"] == "assistant":
            parts = []
            for block in content:
                if block.type == "text":
                    parts.append(types.Part(text=block.text))
                elif block.type == "tool_use":
                    parts.append(types.Part(
                        function_call=types.FunctionCall(name=block.name, args=block.input, id=block.id)
                    ))
                    id_to_name[block.id] = block.name
            contents.append(types.Content(role="model", parts=parts))
            continue

        # role == "user" with list content: tool_result observations fed back to the model.
        parts = []
        for result in content:
            tool_name = id_to_name.get(result["tool_use_id"], "unknown_tool")
            response = {"error": result["content"]} if result["is_error"] else {"output": result["content"]}
            parts.append(types.Part(
                function_response=types.FunctionResponse(name=tool_name, id=result["tool_use_id"], response=response)
            ))
        contents.append(types.Content(role="user", parts=parts))

    return contents, id_to_name


def _translate_response(raw) -> GeminiResponse:
    """A raw Gemini GenerateContentResponse -> the duck-typed GeminiResponse Agent expects."""
    parts = raw.candidates[0].content.parts or []
    blocks: list = []
    has_tool_call = False

    for i, part in enumerate(parts):
        if part.function_call is not None:
            has_tool_call = True
            call_id = part.function_call.id or f"call_{i}"
            blocks.append(ToolUseBlock(
                name=part.function_call.name,
                tool_input=dict(part.function_call.args or {}),
                block_id=call_id,
            ))
        elif part.text:
            blocks.append(TextBlock(part.text))

    stop_reason = "tool_use" if has_tool_call else "end_turn"
    return GeminiResponse(stop_reason=stop_reason, content=blocks)
