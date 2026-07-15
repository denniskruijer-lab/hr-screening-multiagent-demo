"""Factory for the real Anthropic client.

Kept as a thin, separate module (rather than importing `anthropic` directly in
main.py or agent.py) so the rest of the codebase never needs the SDK installed
to be imported -- only to actually run against the real API.
"""
from __future__ import annotations

import os


def build_anthropic_client():
    """Construct a real Anthropic client from the ANTHROPIC_API_KEY environment variable."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Get a key at https://console.anthropic.com "
            "and set it as an environment variable before running against the real API."
        )

    import anthropic  # imported lazily -- keeps this module importable without the SDK installed

    return anthropic.Anthropic()
