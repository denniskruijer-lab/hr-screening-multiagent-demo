"""
Tools for the HR screening use case.

Each handler returns a string (the observation the model sees) and raises
ValueError on invalid input — the framework turns that into an error
observation so the model can correct itself.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"


def read_document(filename: str) -> str:
    """Read a text document from the data/ directory (and nowhere else)."""
    path = (DATA_DIR / filename).resolve()
    # Path-traversal guard: the resolved path must stay inside data/.
    if DATA_DIR.resolve() not in path.parents and path != DATA_DIR.resolve():
        raise ValueError(f"Access outside the data directory is not allowed: {filename}")
    if not path.is_file():
        available = ", ".join(p.name for p in DATA_DIR.glob("*") if p.is_file())
        raise ValueError(f"File '{filename}' not found. Available: {available}")
    return path.read_text(encoding="utf-8")


def get_talent_matrix() -> str:
    """Return the competence matrix (levels x competencies) as JSON."""
    return (DATA_DIR / "talent_matrix.json").read_text(encoding="utf-8")


def save_screening_report(
    candidate_name: str,
    fit_score: int,
    estimated_level: str,
    strengths: list,
    gaps: list,
    advice: str,
) -> str:
    """
    Persist the structured screening result. Strict validation here is the
    guard against hallucinated or incomplete output: if the model invents a
    score of 140 or forgets the gaps, the save is refused with a clear error.
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

    OUTPUT_DIR.mkdir(exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in candidate_name.lower())
    path = OUTPUT_DIR / f"report_{safe_name}.json"
    report = {
        "candidate_name": candidate_name,
        "fit_score": fit_score,
        "estimated_level": estimated_level,
        "strengths": strengths,
        "gaps": gaps,
        "advice": advice,
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"Report saved to {path}"
