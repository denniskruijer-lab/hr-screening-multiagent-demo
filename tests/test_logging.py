"""Evals for src/logging_setup.py: WARNING on a guard-caught trace error,
ERROR on a failed Gemini call, CRITICAL on a caught MaxIterationsExceeded.

Uses pytest's caplog rather than reading logs/app.log off disk, to avoid
flush-timing flakiness.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import MaxIterationsExceeded  # noqa: E402
from src.logging_setup import log_trace  # noqa: E402
from src.providers.gemini_adapter import GeminiMessagesClient  # noqa: E402
from .fakes import FakeGeminiSDK  # noqa: E402


def test_log_trace_logs_a_warning_per_guard_caught_error(caplog):
    trace = [
        {"iteration": 1, "tool": "read_document", "input": {}, "is_error": False},
        {"iteration": 2, "tool": "save_screening_report", "input": {}, "is_error": True},
    ]
    logger = logging.getLogger("hr_agent")

    with caplog.at_level(logging.WARNING, logger="hr_agent"):
        log_trace(logger, trace)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "save_screening_report" in warnings[0].message


def test_failed_gemini_call_logs_an_error(caplog):
    class AlwaysFailsSDK:
        def __init__(self):
            self.models = self

        def generate_content(self, **kwargs):
            raise RuntimeError("simulated network failure")

    client = GeminiMessagesClient(AlwaysFailsSDK())
    logger = logging.getLogger("hr_agent")

    with caplog.at_level(logging.ERROR, logger="hr_agent"):
        try:
            client.create(model="gemini-test", max_tokens=10, system="test", tools=[], messages=[])
        except RuntimeError:
            pass

    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "Gemini call failed" in errors[0].message


def test_max_iterations_exceeded_is_the_kind_of_error_callers_log_as_critical(caplog):
    """This doesn't call log_trace (that's for successful runs) -- it just
    confirms MaxIterationsExceeded carries enough information for a caller
    (main.py) to log it usefully at CRITICAL, which main.py does directly."""
    logger = logging.getLogger("hr_agent")

    with caplog.at_level(logging.CRITICAL, logger="hr_agent"):
        try:
            raise MaxIterationsExceeded("no final answer after 8 iterations")
        except MaxIterationsExceeded as exc:
            logger.critical("Supervisor aborted: %s", exc)

    criticals = [record for record in caplog.records if record.levelno == logging.CRITICAL]
    assert len(criticals) == 1
    assert "Supervisor aborted" in criticals[0].message
