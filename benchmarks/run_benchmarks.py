"""Benchmark runner CLI — one command, one comparison table.

Usage:
    uv run python -m benchmarks.run_benchmarks               # compare vs references
    uv run python -m benchmarks.run_benchmarks --update-refs # (re)write references
    uv run python -m benchmarks.run_benchmarks --only sick,cmc
    uv run python -m benchmarks.run_benchmarks --tabpfn      # include TabPFN baseline

Exit code 1 if any dataset FAILs its reference comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import structlog

from benchmarks.compare import (
    REFERENCE_PATH,
    compare_scores,
    load_references,
    render_table,
    save_references,
)
from benchmarks.datasets import ensure_dataset
from benchmarks.manifest import BENCHMARKS, BenchmarkDataset
from mlteam.observability import configure_logging
from mlteam.orchestration import run_tabular_pipeline
from mlteam.store import GuardedArtifactStore, LocalArtifactStore

_log = structlog.get_logger(__name__)

SEED = 42
RESULTS_PATH = Path(__file__).parent / "last_results.json"


def run_one(entry: BenchmarkDataset, include_tabpfn: bool) -> dict[str, float | str]:
    """Run the deterministic pipeline on one benchmark in an isolated store."""
    data_path = ensure_dataset(entry)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"bench-{entry.name}-") as tmp:
        root = Path(tmp) / "store"
        store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
        result = run_tabular_pipeline(
            store,
            data_path,
            target_column=entry.target_column,
            task_type=entry.task_type,
            primary_metric=entry.primary_metric,
            seed=SEED,
            include_tabpfn=include_tabpfn,
        )
    report = result.evaluation_report
    primary = next(m for m in report.cv_metrics if m.name == entry.primary_metric)
    return {
        "metric": entry.primary_metric,
        "mean": primary.mean,
        "std": primary.std,
        "best_template": result.best_template_id,
        "holdout_untouched": report.holdout_untouched,
        "leakage_all_pass": all(c.passed for c in report.leakage_checks),
        "wall_seconds": round(time.perf_counter() - started, 2),
    }


def main(
    argv: list[str] | None = None,
    *,
    refs_path: Path | None = None,
    results_path: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="mlteam benchmark regression suite")
    parser.add_argument("--update-refs", action="store_true", help="rewrite reference scores")
    parser.add_argument("--only", default="", help="comma-separated benchmark names")
    parser.add_argument("--tabpfn", action="store_true", help="include the TabPFN baseline")
    args = parser.parse_args(argv)
    refs_path = refs_path if refs_path is not None else REFERENCE_PATH
    results_path = results_path if results_path is not None else RESULTS_PATH

    configure_logging(json_output=False)
    selected = tuple(b for b in BENCHMARKS if not args.only or b.name in args.only.split(","))
    if not selected:
        print(f"no benchmarks match --only={args.only!r}", file=sys.stderr)
        return 2

    scores: dict[str, dict[str, float | str]] = {}
    for entry in selected:
        _log.info("benchmark_running", name=entry.name, task=entry.task_type.value)
        scores[entry.name] = run_one(entry, include_tabpfn=args.tabpfn)
        _log.info("benchmark_done", name=entry.name, **scores[entry.name])

    results_path.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")

    # ALWAYS compare against the PREVIOUS references — updating first would
    # compare fresh scores against themselves and hide real drift.
    previous = load_references(refs_path)
    rows = compare_scores(scores, previous)
    print()
    print(render_table(rows))

    unsafe = [
        n for n, s in scores.items() if not (s["holdout_untouched"] and s["leakage_all_pass"])
    ]
    if unsafe:
        print(f"\nSAFETY FAILURE (holdout/leakage): {unsafe}", file=sys.stderr)
        return 1  # never persist references from a safety-failing run

    if args.update_refs:
        previous.update(scores)
        save_references(previous, refs_path)
        print(f"\nreference scores updated for: {', '.join(scores)}")

    if any(row.status == "FAIL" for row in rows):
        print("\nREGRESSION: current scores drifted beyond tolerance", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
