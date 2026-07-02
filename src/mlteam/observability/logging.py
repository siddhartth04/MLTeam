"""Structured JSON logging via structlog.

Every log line is a JSON object with timestamp, level, event, and bound
context — machine-parseable for the dashboard's trace viewer.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: int = logging.INFO, json_output: bool = True) -> None:
    """Configure structlog for the whole process.

    ``json_output=False`` switches to a human-friendly console renderer,
    useful for local debugging; production/dashboard paths keep JSON.
    """
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
