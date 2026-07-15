# CLAUDE.md — project context

Minimal agentic framework (`src/agent.py`) extended into a two-agent,
two-provider, RAG-grounded HR screening pipeline (`src/agents/`,
`src/providers/`, `src/rag/`, `src/tools/`). Owner is learning agentic
engineering; keep everything explainable.

## Conventions
1. `src/agent.py` stays small (< ~180 lines), framework-free (no LangChain
   etc.), and untouched by new features — extend around it, not inside it.
2. The LLM client is always injected; nothing in `src/agent.py` may import a
   provider SDK directly. Provider-specific code lives in `src/providers/`.
3. Every guard gets a test. New failure mode = new eval first. Every client
   in the test suite is a scripted fake (see `tests/fakes.py`) — no test may
   make a real network call.
4. Tools raise `ValueError` on invalid input; never validate inside the
   agent loop. Deterministic logic (e.g. comparing two scores) is a plain
   function, not something left to an LLM's judgement.
5. Code and comments in English; sample data may be Dutch. `data/` is
   fictional and committed; `data_local/` is real and gitignored — never
   commit real personal or client data.
6. Never commit API keys; `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` (or
   `GOOGLE_API_KEY`) come from the environment.
7. New multi-agent patterns should be weighed against
   `docs/architecture-alternatives.md` before adding a new orchestration
   primitive — reuse the existing `Agent`/`Tool` abstraction if at all possible.

## Commands
- Run:        `python -m src.main`
- Chat UI:    `streamlit run streamlit_app.py`
- Evals:      `pytest -q`
