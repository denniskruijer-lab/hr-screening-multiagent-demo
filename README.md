# HR Screening Agent — a minimal agentic framework

A tool-using AI agent built **from scratch** in ~150 lines of Python, plus an
HR use case on top: screen a candidate CV against a vacancy, calibrated on a
competence matrix, and produce a validated, structured report.

Built deliberately without an agent framework: the point is to show the agent
loop itself — LLM call → tool request → guarded execution → observation →
repeat — and the guards that make it reliable, rather than hiding them behind
an abstraction.

## Architecture

```
user prompt
    │
    ▼
┌───────────────────────── Agent.run() ─────────────────────────┐
│  loop (max_iterations ceiling — anti-looping guard):          │
│    1. client.messages.create(system, tools, messages)         │
│    2. stop_reason != "tool_use"  →  return final answer       │
│    3. for each tool_use block:                                │
│         • unknown tool?        → error observation            │
│         • missing arguments?   → error observation            │
│         • handler raises?      → error observation            │
│         • else                 → execute, return output       │
│    4. append observations, continue loop                      │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
output/report_<candidate>.json   (schema-validated before saving)
```

Tools in the HR use case (`src/tools.py`):

| Tool | Purpose | Guard it demonstrates |
|---|---|---|
| `read_document` | Read vacancy / CV from `data/` | Path-traversal protection |
| `get_talent_matrix` | Competence matrix for level calibration | — |
| `save_screening_report` | Persist the structured result | Strict field validation → refuses hallucinated or incomplete output |

## Reliability guards, mapped to failure modes

| Agent failure mode | Guard in this repo | Verified by |
|---|---|---|
| Looping (agent never finishes) | `max_iterations` ceiling raises `MaxIterationsExceeded` | `test_runaway_tool_loop_hits_max_iterations_guard` |
| Incorrect tool use (invented tool) | Unknown-tool check, error fed back so the model self-corrects | `test_unknown_tool_is_fed_back_as_correctable_error` |
| Incorrect tool use (bad arguments) | Required-argument validation *before* execution | `test_missing_required_argument_is_rejected_before_execution` |
| Hallucinated / unstructured output | `save_screening_report` validates every field (score 0–100, level enum, non-empty lists) | tool-level `ValueError`s |
| Crashing tools | Exceptions become observations, never crashes | `test_crashing_tool_becomes_observation_not_crash` |
| Opaque behaviour | Every tool call recorded in a trace, printed after each run | `test_happy_path_terminates_with_final_answer` |

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # console.anthropic.com

python -m src.main                          # uses the sample data in data/
python -m src.main --vacancy vacature.txt --cv kandidaat_cv.txt
```

Output: a screening summary in the terminal, the tool-call trace, and a
validated JSON report in `output/`.

## Run the evals (no API key needed)

```bash
pytest -q
```

The test suite injects a scripted fake client, so the loop's behaviour —
termination, loop guard, error feedback — is verified deterministically and
for free, without a single network call.

## Honest scope

- Sample vacancy, CV and matrix are **fictional**.
- This is a learning/demo project: no retries with backoff, no streaming, no
  parallel tool execution, no persistence layer. Each would be a natural
  next iteration.
- A screening agent **advises**; hiring decisions stay with humans.

## Ideas for extension (good Claude Code exercises)

1. Add a `read_docx` tool (python-docx) so real CV files work.
2. Add an eval asserting the agent reads *both* documents before saving.
3. Add a scoring rubric tool and compare its output with the model's own score.
4. Swap the model or add `--model` as a CLI flag.
