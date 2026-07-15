"""Tools for reading input documents (vacancy, CV, talent matrix).

Each handler returns a string (the observation the model sees) and raises
ValueError on invalid input -- the framework turns that into an error
observation so the model can correct itself.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


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
