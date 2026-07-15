"""Tools and pure helpers for producing and reconciling screening reports.

save_screening_report is a tool (called by an agent, backed by an LLM).
flag_disagreement is a plain function, not a tool -- the arithmetic that
decides whether two scores disagree should not be left to a model, so the
supervisor calls it directly rather than exposing it as something an LLM
could get wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

VALID_SOURCES = {"screening", "calibration"}


def _safe_filename(candidate_name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in candidate_name.lower())


def report_path(candidate_name: str, source: str) -> Path:
    return OUTPUT_DIR / f"report_{_safe_filename(candidate_name)}_{source}.json"


def load_report(candidate_name: str, source: str) -> dict:
    """Read back a previously saved report -- used by the supervisor to recover
    the numeric fit_score a worker agent saved, without re-asking the model for it."""
    path = report_path(candidate_name, source)
    if not path.is_file():
        raise ValueError(f"No saved report found for '{candidate_name}' (source={source}).")
    return json.loads(path.read_text(encoding="utf-8"))


def save_screening_report(
    candidate_name: str,
    fit_score: int,
    estimated_level: str,
    strengths: list,
    gaps: list,
    advice: str,
    source: str = "screening",
) -> str:
    """
    Persist the structured screening result. Strict validation here is the
    guard against hallucinated or incomplete output: if the model invents a
    score of 140 or forgets the gaps, the save is refused with a clear error.

    `source` distinguishes which agent produced this report (the screening
    agent or the calibration agent), since both save reports for the same
    candidate.
    """
    if not isinstance(candidate_name, str) or not candidate_name.strip():
        raise ValueError("candidate_name must be a non-empty string")
    if not isinstance(fit_score, int) or not 0 <= fit_score <= 100:
        raise ValueError("fit_score must be an integer between 0 and 100")
    if estimated_level not in ("junior", "medior", "senior"):
        raise ValueError("estimated_level must be one of: junior, medior, senior")
    if not isinstance(strengths, list) or not strengths:
        raise ValueError("strengths must be a non-empty list of strings")
    if not isinstance(gaps, list):
        raise ValueError("gaps must be a list of strings (may be empty)")
    if not isinstance(advice, str) or len(advice.strip()) < 20:
        raise ValueError("advice must be a substantive string (>= 20 characters)")
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of: {', '.join(sorted(VALID_SOURCES))}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = report_path(candidate_name, source)
    report = {
        "candidate_name": candidate_name,
        "fit_score": fit_score,
        "estimated_level": estimated_level,
        "strengths": strengths,
        "gaps": gaps,
        "advice": advice,
        "source": source,
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"Report saved to {path}"


def flag_disagreement(screening_score: int, calibration_score: int, threshold: int = 15) -> str:
    """Compare the screening and calibration scores; flag if they diverge too far.

    A deterministic check rather than an LLM judgement call -- two agents
    scoring the same candidate should agree within a reasonable margin, and
    "reasonable" is exactly the kind of arithmetic a model should not be
    trusted to get right consistently.
    """
    difference = abs(screening_score - calibration_score)
    if difference > threshold:
        return (
            f"DISAGREEMENT: screening scored {screening_score}, calibration scored "
            f"{calibration_score} (difference of {difference}, threshold is {threshold}). "
            f"Recommend human review before proceeding."
        )
    return (
        f"Agreement: screening scored {screening_score}, calibration scored "
        f"{calibration_score} (difference of {difference}, within the threshold of {threshold})."
    )
