"""Judgment trap scorecard CLI — human-readable view over the trap suite.

Usage:
    uv run python -m benchmarks.run_judgment

pytest (tests/test_judgment_traps.py) is the actual enforcement: each trap
maps to one test there, and this CLI runs those same tests through pytest's
API and reports each trap's real outcome — PASS, FAIL, or NOT-YET(Task N)
for an expected xfail — rather than re-implementing the pass/fail logic.
Exits non-zero only on a FAIL: an unexpected failure on a must-pass trap,
or a blocked trap whose xfail didn't fire the way its marker says it should
(including an unmarked capability landing, which strict=True turns into a
failure so this scorecard — and the human reading it — cannot miss it).
"""

from __future__ import annotations

import pytest

from benchmarks.judgment_traps import TRAPS

_TEST_FILE = "tests/test_judgment_traps.py"

_TRAP_TEST_NAMES = {
    "constant_target": "test_constant_target_fails_loudly_naming_the_target",
    "leaky_feature": "test_leaky_feature_is_flagged_by_leakage_checks",
    "id_memorization": "test_id_memorization_drops_the_identifier_column",
    "duplicate_rows": "test_duplicate_rows_are_deduped_before_split",
    "placeholder_labels": "test_placeholder_labels_are_coerced_and_named",
    "skewed_price_target": "test_skewed_price_target_gets_a_log1p_transform",
    "high_cardinality_categorical": "test_high_cardinality_categorical_beats_the_dummy_baseline",
    "temporal_drift_disguised": "test_temporal_drift_disguised_warns_that_random_cv_is_unsafe",
}


class _ResultCollector:
    def __init__(self) -> None:
        self.outcomes: dict[str, tuple[str, bool]] = {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call":
            return
        test_name = report.nodeid.rsplit("::", 1)[-1]
        was_xfail = getattr(report, "wasxfail", None) is not None
        self.outcomes[test_name] = (report.outcome, was_xfail)


def _status(trap_name: str, outcome: str, was_xfail: bool, blocked_by: str | None) -> str:
    if blocked_by is None:
        return "PASS" if outcome == "passed" else "FAIL"
    if outcome == "skipped" and was_xfail:
        return f"NOT-YET({blocked_by})"
    return "FAIL"  # either genuinely broken, or the strict xfail unexpectedly passed


def main() -> int:
    collector = _ResultCollector()
    node_ids = [f"{_TEST_FILE}::{_TRAP_TEST_NAMES[trap.name]}" for trap in TRAPS]
    pytest.main(["-q", "--no-header", "-p", "no:cacheprovider", *node_ids], plugins=[collector])

    rows = []
    any_fail = False
    for trap in TRAPS:
        test_name = _TRAP_TEST_NAMES[trap.name]
        outcome, was_xfail = collector.outcomes.get(test_name, ("failed", False))
        status = _status(trap.name, outcome, was_xfail, trap.blocked_by)
        any_fail = any_fail or status == "FAIL"
        rows.append((trap.name, trap.expected, status))

    name_w = max(len(r[0]) for r in rows)
    expected_w = max(len(r[1]) for r in rows)
    print()
    print(f"{'trap':<{name_w}}  {'expected':<{expected_w}}  status")
    print(f"{'-' * name_w}  {'-' * expected_w}  ------")
    for name, expected, status in rows:
        print(f"{name:<{name_w}}  {expected:<{expected_w}}  {status}")
    print()

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
