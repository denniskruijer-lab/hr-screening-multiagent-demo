"""Evals for src/tools/documents.py: generic text extraction across
.txt/.docx/.pdf, the path-traversal guard, and searching data/ + data_local/.

docx.Document and pypdf.PdfReader are monkeypatched with tiny fakes -- same
injection philosophy as the LLM client, no binary fixture files to maintain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools import documents  # noqa: E402


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Point DATA_DIR/DATA_LOCAL_DIR/SEARCH_DIRS at throwaway tmp_path subdirs."""
    data_dir = tmp_path / "data"
    data_local_dir = tmp_path / "data_local"
    data_dir.mkdir()
    data_local_dir.mkdir()
    monkeypatch.setattr(documents, "DATA_DIR", data_dir)
    monkeypatch.setattr(documents, "DATA_LOCAL_DIR", data_local_dir)
    monkeypatch.setattr(documents, "SEARCH_DIRS", (data_dir, data_local_dir))
    return data_dir, data_local_dir


def test_reads_txt_from_data_dir(isolated_dirs):
    data_dir, _ = isolated_dirs
    (data_dir / "vacature.txt").write_text("AI Engineer vacancy", encoding="utf-8")

    assert documents.read_document("vacature.txt") == "AI Engineer vacancy"


def test_reads_from_data_local_dir_too(isolated_dirs):
    _, data_local_dir = isolated_dirs
    (data_local_dir / "real_cv.txt").write_text("Real CV content", encoding="utf-8")

    assert documents.read_document("real_cv.txt") == "Real CV content"


def test_path_traversal_is_rejected(isolated_dirs):
    with pytest.raises(ValueError, match="not found"):
        documents.read_document("../../etc/passwd")


def test_missing_file_raises_a_clear_error(isolated_dirs):
    with pytest.raises(ValueError, match="not found"):
        documents.read_document("nonexistent.txt")


def test_unsupported_extension_raises_a_clear_error(isolated_dirs):
    data_dir, _ = isolated_dirs
    (data_dir / "cv.exe").write_bytes(b"not a document")

    with pytest.raises(ValueError, match="Unsupported file type"):
        documents.read_document("cv.exe")


def test_reads_docx_via_paragraph_extraction(isolated_dirs, monkeypatch):
    """Extraction walks the raw element tree (body.xpath('.//w:p')), not just
    document.paragraphs -- real CV templates often put content in text boxes,
    which document.paragraphs alone would miss entirely."""
    data_dir, _ = isolated_dirs
    (data_dir / "cv.docx").write_bytes(b"not a real docx -- Document() is faked below")

    class FakeRun:
        def __init__(self, text):
            self.text = text

    class FakeParagraphElement:
        def __init__(self, runs):
            self._runs = runs

        def xpath(self, query):
            assert query == ".//w:t"
            return self._runs

    class FakeBody:
        def __init__(self, paragraphs):
            self._paragraphs = paragraphs

        def xpath(self, query):
            assert query == ".//w:p"
            return self._paragraphs

    class FakeElement:
        def __init__(self, body):
            self.body = body

    class FakeDocument:
        def __init__(self, path):
            paragraphs = [
                FakeParagraphElement([FakeRun("Jamie "), FakeRun("Visser")]),
                FakeParagraphElement([FakeRun("Data Engineer")]),
            ]
            self.element = FakeElement(FakeBody(paragraphs))

    monkeypatch.setattr("docx.Document", FakeDocument)

    assert documents.read_document("cv.docx") == "Jamie Visser\nData Engineer"


def test_reads_pdf_via_page_extraction(isolated_dirs, monkeypatch):
    data_dir, _ = isolated_dirs
    (data_dir / "cv.pdf").write_bytes(b"not a real pdf -- PdfReader() is faked below")

    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakePdfReader:
        def __init__(self, path):
            self.pages = [FakePage("Page one text"), FakePage("Page two text")]

    monkeypatch.setattr("pypdf.PdfReader", FakePdfReader)

    assert documents.read_document("cv.pdf") == "Page one text\nPage two text"
