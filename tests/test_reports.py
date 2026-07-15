"""Evals for src/tools/reports.py: the save/validate/reconcile logic that
backs the screening and calibration agents.

Every test writes into a tmp_path, never the real output/ directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools import reports  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "OUTPUT_DIR", tmp_path)


def _valid_report_kwargs(**overrides):
    kwargs = dict(
        candidate_name="Jamie Visser",
        fit_score=72,
        estimated_level="medior",
        strengths=["Python", "RAG side-projects"],
        gaps=["No production agent experience"],
        advice="Recommend a technical interview focused on agent reliability.",
    )
    kwargs.update(overrides)
    return kwargs


def test_save_screening_report_defaults_source_to_screening():
    message = reports.save_screening_report(**_valid_report_kwargs())
    assert "screening.json" in message


def test_save_screening_report_accepts_calibration_source():
    message = reports.save_screening_report(**_valid_report_kwargs(), source="calibration")
    assert "calibration.json" in message


def test_save_screening_report_rejects_invalid_source():
    with pytest.raises(ValueError, match="source must be one of"):
        reports.save_screening_report(**_valid_report_kwargs(), source="made_up_source")


def test_save_screening_report_rejects_out_of_range_score():
    with pytest.raises(ValueError, match="fit_score"):
        reports.save_screening_report(**_valid_report_kwargs(fit_score=140))


def test_load_report_round_trips_the_saved_score():
    reports.save_screening_report(**_valid_report_kwargs(fit_score=64), source="screening")
    loaded = reports.load_report("Jamie Visser", source="screening")
    assert loaded["fit_score"] == 64


def test_load_report_raises_a_clear_error_when_nothing_was_saved():
    with pytest.raises(ValueError, match="No saved report found"):
        reports.load_report("Nobody Home", source="screening")


def test_flag_disagreement_flags_when_scores_diverge():
    message = reports.flag_disagreement(screening_score=80, calibration_score=55)
    assert message.startswith("DISAGREEMENT")


def test_flag_disagreement_agrees_when_scores_are_close():
    message = reports.flag_disagreement(screening_score=80, calibration_score=72)
    assert message.startswith("Agreement")
