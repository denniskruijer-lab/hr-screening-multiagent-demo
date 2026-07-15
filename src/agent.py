"""
Minimal agentic framework: one explicit loop, injected LLM client, hard guards.

Design goals (in order):
1. Understandable — the whole agent fits in one file you can explain line by line.
2. Testable — the client is injected, so behaviour is verified without network calls.
3. Reliable — every known failure mode of tool-using agents has an explicit guard:
   - looping            -> max_iterations ceiling (MaxIterationsExceeded)
   - incorrect tool use -> unknown-tool + missing-argument validation, fed back
                           to the model as an error observation so it can self-correct
   - crashing tools     -> exceptions become error observations, never crashes
   - opaque behaviour   -> every tool call is recorded in a trace
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class MaxIterationsExceeded(Exception):
    """Raised when the agent keeps calling tools without producing a final answer."""


@dataclass
class Tool:
    """A capability the model may invoke. Handler receives the tool input as kwargs."""
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., str]

    def spec(self) -> dict:
        """The tool definition in the shape the Anthropic Messages API expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class AgentResult:
    final_text: str
    iterations: int
    trace: list[dict] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        client: Any,
        model: str,
        tools: list[Tool],
        system_prompt: str,
        max_iterations: int = 8,
        max_tokens: int = 2000,
    ):
        self.client = client
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens

    # ── the agent loop ──────────────────────────────────────────────────────
    def run(self, user_prompt: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": user_prompt}]
        trace: list[dict] = []

        for iteration in range(1, self.max_iterations + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                tools=[t.spec() for t in self.tools.values()],
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            # Model is done: collect the text blocks and return.
            if response.stop_reason != "tool_use":
                final_text = "".join(
                    block.text
                    for block in response.content
                    if getattr(block, "type", "") == "text"
                )
                return AgentResult(final_text=final_text, iterations=iteration, trace=trace)

            # Model wants tools: execute each request, feed observations back.
            tool_results = []
            for block in response.content:
                if getattr(block, "type", "") != "tool_use":
                    continue
                output, is_error = self._execute(block.name, block.input or {})
                trace.append({
                    "iteration": iteration,
                    "tool": block.name,
                    "input": block.input,
                    "is_error": is_error,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": tool_results})

        raise MaxIterationsExceeded(
            f"No final answer after {self.max_iterations} iterations — possible tool loop. "
            f"Trace: {[t['tool'] for t in trace]}"
        )

    # ── guarded tool execution ──────────────────────────────────────────────
    def _execute(self, name: str, args: dict) -> tuple[str, bool]:
        # Guard: model invented a tool that does not exist.
        tool = self.tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self.tools))
            return f"Unknown tool '{name}'. Available tools: {available}.", True

        # Guard: required arguments missing (schema-level validation before execution).
        required = tool.input_schema.get("required", [])
        missing = [key for key in required if key not in args]
        if missing:
            return (
                f"Missing required argument(s) {missing} for tool '{name}'. "
                f"Expected schema: {tool.input_schema}", True,
            )

        # Guard: a crashing tool becomes an observation, not a crash.
        try:
            return str(tool.handler(**args)), False
        except Exception as exc:  # noqa: BLE001 — deliberate: feed error back to model
            return f"Tool '{name}' failed: {exc}", True
