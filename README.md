# HR Screening Multi-Agent Demo

A tool-using AI agent framework built **from scratch** (`src/agent.py`, ~130
lines, unchanged since the very first version of this project), extended
into a two-agent, two-provider, RAG-grounded screening pipeline: screen a
candidate CV against a vacancy, independently double-check the score on a
different LLM provider, ground both in retrieved context, and produce a
validated, structured report.

Built deliberately without an agent framework: the point is to show the
agent loop itself — LLM call → tool request → guarded execution →
observation → repeat — and the guards that make it reliable, rather than
hiding them behind an abstraction. Every extension below (a second provider,
a supervisor, RAG, real document formats, a UI) was added *around* that
loop, not by changing it — `src/agent.py` has zero diff from the very first
commit in this repo's history.

## Architecture

```
user prompt
    │
    ▼
┌───────────────────── Supervisor (Gemini Flash) ──────────────────────┐
│  1. run_screening_agent      -> Screening Agent (Claude Sonnet)      │
│  2. run_calibration_agent    -> Calibration Agent (Gemini Pro)       │
│  3. check_score_agreement    -> flag_disagreement() (pure function)  │
└────────────────────────────────────────────────────────────────────────┘
    │                                    │
    ▼                                    ▼
Screening Agent                   Calibration Agent
(reads docs, retrieves context,   (independently re-scores the
 saves report_<name>_screening)    same candidate, on a different
                                    provider on purpose, saves
                                    report_<name>_calibration)
```

Each of the three boxes above is the **same** `Agent` class
(`src/agent.py`) — the supervisor is not a new orchestration primitive,
it's a plain agent whose "tools" happen to be other agents' `.run()`
methods. See `docs/architecture-alternatives.md` for the two other
multi-agent patterns considered (network, hierarchical) and why supervisor
fits this problem instead.

Agent loop (unchanged since the first commit):

```
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
```

### Multi-provider routing (`src/providers/`)

`Agent` only requires its `client` to expose `client.messages.create(...)`
returning an object with `.stop_reason` and `.content` — that's the real
Anthropic client's shape, but it's duck-typed, not an Anthropic class.
`src/providers/gemini_adapter.py::GeminiMessagesClient` satisfies the same
shape by translating to and from the Google GenAI SDK, so the screening
agent (Claude) and the calibration agent (Gemini) are both plain `Agent`
instances — no forking, no per-provider subclass.

Model choice is a function of each agent's role, not a fixed default:

| Agent | Model | Why |
|---|---|---|
| Supervisor | `gemini-2.5-flash` | Cheap and fast — it only dispatches and aggregates, it doesn't reason deeply. |
| Screening | `claude-sonnet-5` | The primary, deep CV-vs-vacancy assessment. |
| Calibration | `gemini-2.5-pro` | A different **provider** on purpose — a second opinion is only useful if it isn't correlated with the first model's blind spots. |

### RAG (`src/rag/`, `src/tools/retrieval.py`)

Both worker agents get a `retrieve_context` tool backed by
`InMemoryVectorStore`: the vacancy, CV, and talent matrix are chunked and
embedded once (Gemini's embedding API) at startup, and retrieval is a
hand-rolled cosine similarity search — no vector database needed at this
scale. If the embedding call fails, the store degrades to naive keyword
overlap rather than breaking (same "crashing tools become observations"
philosophy as the rest of this codebase, one layer up).

### Reading real documents (`src/tools/documents.py`)

`read_document` extracts text from `.txt`, `.docx`, and `.pdf` — generic
text *extraction* across common file **formats**, not generic CV *structure*
parsing (tables, columns, scanned layouts). The LLM interprets the extracted
text; this tool's only job is getting text out of whatever container it's
in. It also searches `data/` (fictional, committed) and `data_local/`
(real, gitignored) so the same code path handles the sample data and a real
CV/vacancy pair.

## Reliability guards, mapped to failure modes

| Agent failure mode | Guard in this repo | Verified by |
|---|---|---|
| Looping (agent never finishes) | `max_iterations` ceiling raises `MaxIterationsExceeded` | `test_runaway_tool_loop_hits_max_iterations_guard` |
| Incorrect tool use (invented tool) | Unknown-tool check, error fed back so the model self-corrects | `test_unknown_tool_is_fed_back_as_correctable_error` |
| Incorrect tool use (bad arguments) | Required-argument validation *before* execution | `test_missing_required_argument_is_rejected_before_execution` |
| Hallucinated / unstructured output | `save_screening_report` validates every field (score 0-100, level enum, non-empty lists) | tool-level `ValueError`s, `tests/test_reports.py` |
| Crashing tools | Exceptions become observations, never crashes | `test_crashing_tool_becomes_observation_not_crash` |
| Failed provider/embedding calls | Re-raised clearly (provider) or degraded gracefully (embeddings) rather than corrupting agent state | `tests/test_gemini_adapter.py`, `tests/test_rag.py` |
| Two agents disagreeing | `flag_disagreement()` — deterministic arithmetic, not left to an LLM's judgement | `tests/test_reports.py` |
| Opaque behaviour | Every tool call recorded in a trace; WARNING/ERROR/CRITICAL logged to `logs/app.log` | `tests/test_logging.py` |

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...          # console.anthropic.com
export GEMINI_API_KEY=...                    # aistudio.google.com/apikey

python -m src.main                            # uses the fictional sample data in data/
python -m src.main --cv kandidaat_cv.txt --vacancy vacature.txt --candidate-name "Jamie Visser"
```

Or run the chat UI:

```bash
streamlit run streamlit_app.py
```

Output: a screening + calibration summary, the agreement check, the full
tool-call trace, and validated JSON reports in `output/`
(`report_<candidate>_screening.json` and `report_<candidate>_calibration.json`).

## Run the evals (no API key needed)

```bash
pytest -q
```

Every client in this test suite — Anthropic, Gemini, and the embeddings
call — is a scripted fake (`tests/fakes.py`), so the loop's behaviour,
both adapters, the supervisor's dispatch/reconciliation, RAG's ranking and
degradation, and the logging paths are all verified deterministically,
without a single network call. This is also why CI (`.github/workflows/ci.yml`)
needs no API keys or secrets to run on every push.

## Honest scope

- Sample vacancy, CV, and matrix (`data/`) are **fictional**. Real data
  (`data_local/`) is gitignored and used only for local demo runs.
- Generic file-format support (`.txt`/`.docx`/`.pdf`) is text extraction,
  not CV-structure parsing — a much harder, open-ended problem that this
  project doesn't attempt (see `src/tools/documents.py`).
- RAG uses Gemini's embedding API by choice, which means retrieval depends
  on network availability; the store degrades to keyword overlap rather
  than crashing if that call fails, but that's a real tradeoff, not a
  non-issue.
- This is a learning/demo project: no retries with backoff, no streaming,
  no parallel tool execution. Each would be a natural next iteration.
- A screening agent **advises**; hiring decisions stay with humans.

## Project structure

```
src/
  agent.py              # the core loop -- unchanged since the first commit
  logging_setup.py       # WARNING/ERROR/CRITICAL to logs/app.log
  main.py                 # CLI entry point: builds and runs the supervisor
  providers/               # Anthropic + Gemini client factories, the Gemini adapter
  rag/                      # embeddings, in-memory vector store
  tools/                     # read_document (+docx/pdf), reports, retrieval
  agents/                     # screening, calibration, and the supervisor
streamlit_app.py            # thin chat UI over the same supervisor
docs/architecture-alternatives.md   # why supervisor, not network or hierarchical
```

## Ideas for extension (good next iterations)

1. Add retries with backoff around provider calls.
2. Add a scoring rubric tool and compare its output with the model's own score.
3. Add streaming responses to the CLI/UI.
4. Swap the vector store for a real vector database if the corpus grows past a few documents.
