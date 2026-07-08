"""Lessons substrate: cross-run distilled propositions, as a PRIOR only.

Mirrors ledger.py's shape exactly: an append-only JSONL file at
``store_root / "lessons.jsonl"``, one entry per distilled lesson, written
best-effort and never raising. Where the ledger records WHICH template won,
this records WHAT a run taught ("log-transform price-like targets") -- the
founding incident's lesson had nowhere to live before this, so every
failure was re-litigated from scratch.

Consumed as a prior, never a rule: planner/cleaner prompts frame these as
"weigh, don't blindly obey" (agents/retro.py, agents/experiment_planner.py),
matching the ledger's own priors-may-reorder-never-skip-coverage discipline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from mlteam.contracts.base import utcnow

_LESSONS_FILENAME = "lessons.jsonl"
_MAX_LESSON_CHARS = 300


class LessonEntry(TypedDict):
    lesson: str
    evidence: str
    task_type: str
    source_run_id: str
    timestamp: str


def record_lesson(
    store_root: Path,
    *,
    lesson: str,
    evidence: str,
    task_type: str,
    source_run_id: str,
) -> LessonEntry:
    """Append one distilled lesson and return the entry written. Never
    raises -- a broken lessons write must not fail an otherwise-successful
    run (callers should wrap this in their own best-effort try/except, same
    as store/ledger.py's record_outcome).

    ``lesson`` over ``_MAX_LESSON_CHARS`` is truncated, not rejected: the
    caller (agents/retro.py) already wire-validates length via the LLM
    schema, so a violation here means the schema's own cap slipped, not
    that this run's real lesson should be silently dropped.
    """
    entry: LessonEntry = {
        "lesson": lesson[:_MAX_LESSON_CHARS],
        "evidence": evidence,
        "task_type": task_type,
        "source_run_id": source_run_id,
        "timestamp": utcnow().isoformat(timespec="seconds"),
    }
    path = store_root / _LESSONS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_lessons(store_root: Path) -> list[LessonEntry]:
    """All recorded lessons, oldest first. Empty list if none recorded yet."""
    path = store_root / _LESSONS_FILENAME
    if not path.exists():
        return []
    entries: list[LessonEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def matching_lessons(
    store_root: Path,
    *,
    task_type: str,
    max_entries: int = 5,
) -> list[LessonEntry]:
    """Up to ``max_entries`` most recent lessons for ``task_type``, most
    recent first -- the same shape as ledger.matching_priors."""
    matches = [e for e in read_lessons(store_root) if e["task_type"] == task_type]
    return list(reversed(matches))[:max_entries]
