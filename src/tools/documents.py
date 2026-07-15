"""Tools for reading input documents (vacancy, CV, talent matrix).

Each handler returns a string (the observation the model sees) and raises
ValueError on invalid input -- the framework turns that into an error
observation so the model can correct itself.

read_document extracts plain text from whatever common file format it's
handed (.txt, .docx, .pdf). It does NOT try to understand CV *structure* --
tables, columns, section headers -- that is a much harder, open-ended
problem (and the entire business of commercial resume-parsing products).
The LLM interprets the extracted text; this tool's only job is getting text
out of a container.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_LOCAL_DIR = PROJECT_ROOT / "data_local"
SEARCH_DIRS = (DATA_DIR, DATA_LOCAL_DIR)


def _resolve_within_search_dirs(filename: str) -> Path:
    """Find `filename` inside one of SEARCH_DIRS, guarding against path traversal.

    Each candidate directory is checked independently: the resolved path must
    stay inside that specific directory, so "../../etc/passwd"-style
    filenames can't escape into the rest of the filesystem via either dir.
    """
    for base_dir in SEARCH_DIRS:
        candidate = (base_dir / filename).resolve()
        base_resolved = base_dir.resolve()
        within_base_dir = base_resolved in candidate.parents or candidate == base_resolved
        if within_base_dir and candidate.is_file():
            return candidate

    available = ", ".join(
        p.name for base_dir in SEARCH_DIRS for p in base_dir.glob("*") if p.is_file()
    )
    raise ValueError(f"File '{filename}' not found. Available: {available}")


def read_document(filename: str) -> str:
    """Read a document (.txt, .docx, or .pdf) from data/ or data_local/."""
    path = _resolve_within_search_dirs(filename)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".pdf":
        return _read_pdf(path)

    raise ValueError(f"Unsupported file type '{suffix}' for '{filename}'. Supported: .txt, .docx, .pdf")


def _read_docx(path: Path) -> str:
    """Extract all paragraph text, including text inside text boxes.

    python-docx's high-level `document.paragraphs` only walks the main body --
    many real-world CV templates (this project's own real CV included) put
    all their content inside floating text boxes for layout, which live
    elsewhere in the document's XML tree. Walking the raw element tree with
    xpath reaches paragraphs wherever they actually are.
    """
    from docx import Document  # imported lazily -- keeps this module importable without the package installed

    document = Document(path)
    paragraphs = document.element.body.xpath(".//w:p")
    texts = ["".join(run.text or "" for run in paragraph.xpath(".//w:t")) for paragraph in paragraphs]
    return "\n".join(text for text in texts if text.strip())


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily -- keeps this module importable without the package installed

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def get_talent_matrix() -> str:
    """Return the competence matrix (levels x competencies) as JSON."""
    return (DATA_DIR / "talent_matrix.json").read_text(encoding="utf-8")
