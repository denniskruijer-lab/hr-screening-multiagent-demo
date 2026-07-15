"""
CLI entry point: screen a candidate CV against a vacancy using the agent.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m src.main
    python -m src.main --vacancy vacature.txt --cv kandidaat_cv.txt
"""
from __future__ import annotations

import argparse
import os
import sys

from .agent import Agent, MaxIterationsExceeded, Tool
from . import tools

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a recruitment screening agent for a technology consultancy.

Your job: assess how well a candidate fits a vacancy, calibrated against the
company's talent matrix.

Work step by step:
1. Read the vacancy and the candidate CV with read_document.
2. Fetch the talent matrix with get_talent_matrix.
3. Compare the candidate against both the vacancy requirements and the matrix.
4. Save your conclusion with save_screening_report, then summarise it briefly.

Ground every claim in the documents you actually read. If information is
missing, say so in the report instead of guessing."""


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="read_document",
            description="Read a text document (vacancy or CV) from the data directory.",
            input_schema={
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "File name inside data/"}},
                "required": ["filename"],
            },
            handler=tools.read_document,
        ),
        Tool(
            name="get_talent_matrix",
            description="Get the competence matrix (levels x competencies) used for calibration.",
            input_schema={"type": "object", "properties": {}},
            handler=tools.get_talent_matrix,
        ),
        Tool(
            name="save_screening_report",
            description="Save the final structured screening report. All fields are validated.",
            input_schema={
                "type": "object",
                "properties": {
                    "candidate_name": {"type": "string"},
                    "fit_score": {"type": "integer", "description": "0-100"},
                    "estimated_level": {"type": "string", "enum": ["junior", "medior", "senior"]},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "gaps": {"type": "array", "items": {"type": "string"}},
                    "advice": {"type": "string", "description": "Concrete next step, >= 20 chars"},
                },
                "required": ["candidate_name", "fit_score", "estimated_level", "strengths", "gaps", "advice"],
            },
            handler=tools.save_screening_report,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen a candidate CV against a vacancy.")
    parser.add_argument("--vacancy", default="vacature.txt")
    parser.add_argument("--cv", default="kandidaat_cv.txt")
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Get a key at https://console.anthropic.com "
              "and run: export ANTHROPIC_API_KEY=sk-ant-...")
        return 1

    import anthropic  # imported lazily so tests never need the SDK

    agent = Agent(
        client=anthropic.Anthropic(),
        model=MODEL,
        tools=build_tools(),
        system_prompt=SYSTEM_PROMPT,
        max_iterations=args.max_iterations,
    )

    prompt = (
        f"Screen the candidate in '{args.cv}' against the vacancy in '{args.vacancy}'. "
        f"Use the talent matrix for level calibration and save a report."
    )

    try:
        result = agent.run(prompt)
    except MaxIterationsExceeded as exc:
        print(f"Agent aborted: {exc}")
        return 2

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
