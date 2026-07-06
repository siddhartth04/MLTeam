"""Outcome ledger: cross-run experience for the planner, as a PRIOR only.

An append-only JSONL file recording, per completed run, which template won
and by how much. The planner reads a handful of similar past runs and may
REORDER its candidates and budgets accordingly — it may never use priors to
skip the mandatory baselines or to avoid evaluating a candidate. That
guarantee is enforced in code (ExperimentPlan's baselines-first validator),
not by trusting the LLM to behave; priors can bias ordering, never coverage.

One JSONL file per store root, never rewritten — only appended — so it is
safe to write from a background run thread without any locking beyond the
OS's atomic small-write append guarantee.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TypedDict

from mlteam.contracts.base import utcnow

_LEDGER_FILENAME = "outcome_ledger.jsonl"


class LedgerEntry(TypedDict):
    task_type: str
    n_rows_bucket: int
    n_features: int
    categorical_share: float
    winner_template: str
    margin_vs_second: float | None
    primary_metric: str
    timestamp: str


def _rows_bucket(n_rows: int) -> int:
    """log10 bucket: 100 rows and 900 rows both look like "hundreds" to a
    planner deciding whether e.g. SVM's O(n^2) cost is affordable."""
    return int(math.log10(max(n_rows, 1)))


def record_outcome(
    store_root: Path,
    *,
    task_type: str,
    n_rows: int,
    n_features: int,
    categorical_share: float,
    winner_template: str,
    margin_vs_second: float | None,
    primary_metric: str,
) -> None:
    """Append one completed run's outcome. Never raises — a broken ledger
    write must not fail an otherwise-successful run."""
    entry: LedgerEntry = {
        "task_type": task_type,
        "n_rows_bucket": _rows_bucket(n_rows),
        "n_features": n_features,
        "categorical_share": round(categorical_share, 1),
        "winner_template": winner_template,
        "margin_vs_second": margin_vs_second,
        "primary_metric": primary_metric,
        "timestamp": utcnow().isoformat(timespec="seconds"),
    }
    path = store_root / _LEDGER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_ledger(store_root: Path) -> list[LedgerEntry]:
    """All recorded entries, oldest first. Empty list if none exist yet."""
    path = store_root / _LEDGER_FILENAME
    if not path.exists():
        return []
    entries: list[LedgerEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def matching_priors(
    store_root: Path,
    *,
    task_type: str,
    n_rows: int,
    max_entries: int = 5,
) -> list[LedgerEntry]:
    """Up to ``max_entries`` most recent entries for the same task_type and a
    similar data scale (bucket +/- 1), most recent first."""
    bucket = _rows_bucket(n_rows)
    matches = [
        e
        for e in read_ledger(store_root)
        if e["task_type"] == task_type and abs(e["n_rows_bucket"] - bucket) <= 1
    ]
    return list(reversed(matches))[:max_entries]
