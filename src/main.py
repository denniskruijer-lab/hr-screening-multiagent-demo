"""
CLI entry point: run the two-agent screening pipeline (supervisor + workers)
on a candidate.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export GEMINI_API_KEY=...
    python -m src.main
    python -m src.main --vacancy vacature.txt --cv kandidaat_cv.txt --candidate-name "Jamie Visser"
"""
from __future__ import annotations

import argparse
import sys

from .agent import MaxIterationsExceeded
from .agents.supervisor import build_supervisor
from .logging_setup import configure_logging, log_trace
from .providers.anthropic_client import build_anthropic_client
from .providers.gemini_adapter import GeminiMessagesClient, build_gemini_sdk_client
from .rag.store import InMemoryVectorStore, build_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen a candidate CV against a vacancy using two agents.")
    parser.add_argument("--vacancy", default="vacature.txt")
    parser.add_argument("--cv", default="kandidaat_cv.txt")
    parser.add_argument("--candidate-name", default="Jamie Visser",
                         help="Must match the name the agents save their reports under.")
    args = parser.parse_args()

    logger = configure_logging()

    try:
        anthropic_client = build_anthropic_client()
        gemini_sdk = build_gemini_sdk_client()
    except RuntimeError as exc:
        print(exc)
        return 1

    # One Gemini-backed client, reused for both the calibration agent and the
    # supervisor itself -- the model is chosen per Agent instance, not baked
    # into the client, so sharing it is safe.
    gemini_client = GeminiMessagesClient(gemini_sdk)

    corpus = build_corpus([args.cv, args.vacancy, "talent_matrix.json"])
    vector_store = InMemoryVectorStore(corpus, gemini_sdk)

    supervisor = build_supervisor(
        supervisor_client=gemini_client,
        screening_client=anthropic_client,
        calibration_client=gemini_client,
        vector_store=vector_store,
    )

    prompt = (
        f"Screen the candidate named '{args.candidate_name}' using the CV in '{args.cv}' "
        f"against the vacancy in '{args.vacancy}'."
    )

    try:
        result = supervisor.run(prompt)
    except MaxIterationsExceeded as exc:
        logger.critical("Supervisor aborted: %s", exc)
        print(f"Supervisor aborted: {exc}")
        return 2

    log_trace(logger, result.trace)

    print("=" * 72)
    print(result.final_text)
    print("=" * 72)
    print(f"Iterations: {result.iterations} | Tool calls: {len(result.trace)}")
    for step in result.trace:
        flag = "ERROR" if step["is_error"] else "ok"
        print(f"  [{step['iteration']}] {step['tool']} -> {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
