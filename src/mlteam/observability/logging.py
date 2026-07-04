"""Structured JSON logging via structlog.

Every log line is a JSON object with timestamp, level, event, and bound
context — machine-parseable for the dashboard's trace viewer.
"""

from __future__ import annotations

import contextlib
import logging
import sys

import structlog


def configure_logging(level: int = logging.INFO, json_output: bool = True) -> None:
    """Configure structlog for the whole process.

    ``json_output=False`` switches to a human-friendly console renderer,
    useful for local debugging; production/dashboard paths keep JSON.
    """
    # Windows consoles default to cp1252, which cannot encode Unicode in log
    # messages (e.g. the '·' in gate labels). Force UTF-8 so logging never
    # crashes the process on a stray non-latin1 character.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # Stream may be redirected/non-reconfigurable; best-effort only.
            with contextlib.suppress(ValueError, OSError):  # pragma: no cover
                reconfigure(encoding="utf-8", errors="replace")
    renderer: structlog.typing.Processor
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
