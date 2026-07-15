"""Structured logging: WARNING/ERROR/CRITICAL only, appended to a per-run logfile.

Kept separate from agent.py on purpose -- logging happens at the call sites
around Agent.run() (here, and in the provider adapters), not inside the tested
core loop itself, so agent.py stays untouched.
"""
from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"
LOGGER_NAME = "hr_agent"


def configure_logging() -> logging.Logger:
    """Set up the shared 'hr_agent' logger: WARNING floor, appended to logs/app.log."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger  # already configured elsewhere in this process -- don't duplicate handlers

    LOG_FILE.parent.mkdir(exist_ok=True)
    handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    return logger


def log_trace(logger: logging.Logger, trace: list[dict]) -> None:
    """Log one WARNING per guard-caught error in an agent's tool-call trace."""
    for step in trace:
        if step.get("is_error"):
            logger.warning(
                "Tool guard caught a correctable error: tool=%s input=%s (iteration %s)",
                step["tool"], step["input"], step["iteration"],
            )
