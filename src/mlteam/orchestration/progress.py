"""Live progress reporting: a tiny protocol threaded through the pipeline so
each agent boundary can emit a real stage transition + log line as it happens.

Plain `str` (not the `api.events` enums) so this module has no dependency on
`mlteam.api` — the orchestration layer must not import the API layer. The run
manager maps these strings to `Stage`/`StageStatus` at the boundary.
"""

from __future__ import annotations

from typing import Protocol


class ProgressReporter(Protocol):
    def stage(self, stage: str, status: str, detail: str = "") -> None: ...

    def log(self, message: str, *, level: str = "info", stage: str | None = None) -> None: ...


class NullReporter:
    """Default reporter — does nothing, so non-API callers (tests, demo
    scripts, direct library use) are unaffected by progress reporting."""

    def stage(self, stage: str, status: str, detail: str = "") -> None: ...

    def log(self, message: str, *, level: str = "info", stage: str | None = None) -> None: ...


NULL_REPORTER = NullReporter()
