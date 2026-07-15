# CLAUDE.md — project context

Minimal agentic framework (src/agent.py) + HR screening use case (src/tools.py,
src/main.py). Owner is learning agentic engineering; keep everything explainable.

## Conventions
1. The framework stays small (< ~180 lines) and framework-free — no LangChain etc.
2. The LLM client is always injected; nothing in src/agent.py may import anthropic.
3. Every guard gets a test in tests/test_agent.py. New failure mode = new eval first.
4. Tools raise ValueError on invalid input; never validate inside the agent loop.
5. Code and comments in English; sample data may be Dutch. Fictional data only.
6. Never commit API keys; ANTHROPIC_API_KEY comes from the environment.

## Commands
- Run:   python -m src.main
- Evals: pytest -q
