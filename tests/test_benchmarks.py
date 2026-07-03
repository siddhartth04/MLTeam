"""Benchmark harness tests — no network required."""

from __future__ import annotations

import polars as pl
import pytest
from benchmarks.compare import ComparisonRow, compare_scores, render_table, tolerance_for
from benchmarks.datasets import ensure_dataset, make_temporal_churn_frame
from benchmarks.manifest import BENCHMARKS, get_benchmark

from mlteam.contracts import TaskType


def test_manifest_coverage():
    assert 6 <= len(BENCHMARKS) <= 10
    names = [b.name for b in BENCHMARKS]
    assert len(names) == len(set(names)), "benchmark names must be unique"
    tasks = {b.task_type for b in BENCHMARKS}
    assert tasks == {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.REGRESSION,
    }
    assert any(b.imbalanced for b in BENCHMARKS)
    assert any(b.has_datetime for b in BENCHMARKS)


def test_openml_entries_have_ids():
    for entry in BENCHMARKS:
        if entry.source == "openml":
            assert entry.openml_id is not None, entry.name


def test_get_benchmark_unknown_raises():
    with pytest.raises(KeyError, match="unknown benchmark"):
        get_benchmark("nope")


def test_temporal_frame_has_real_datetime():
    frame = make_temporal_churn_frame(n_rows=50)
    assert frame["signup_date"].dtype == pl.Datetime
    assert frame["churned"].n_unique() == 2


def test_temporal_frame_is_deterministic():
    assert make_temporal_churn_frame(n_rows=50).equals(make_temporal_churn_frame(n_rows=50))


def test_ensure_dataset_caches_synthetic(tmp_path):
    entry = get_benchmark("temporal_churn")
    path = ensure_dataset(entry, data_dir=tmp_path)
    assert path.exists()
    first_mtime = path.stat().st_mtime
    assert ensure_dataset(entry, data_dir=tmp_path) == path
    assert path.stat().st_mtime == first_mtime  # cache hit, not re-fetched


def test_compare_within_tolerance_passes():
    rows = compare_scores(
        {"a": {"metric": "roc_auc", "mean": 0.801}},
        {"a": {"metric": "roc_auc", "mean": 0.80}},
    )
    assert rows[0].status == "PASS"


def test_compare_beyond_tolerance_fails():
    rows = compare_scores(
        {"a": {"metric": "roc_auc", "mean": 0.70}},
        {"a": {"metric": "roc_auc", "mean": 0.80}},
    )
    assert rows[0].status == "FAIL"
    assert rows[0].delta == pytest.approx(-0.10)


def test_compare_missing_reference_is_new():
    rows = compare_scores({"a": {"metric": "rmse", "mean": 3.2}}, {})
    assert rows[0].status == "NEW"


def test_tolerance_scales_with_metric_magnitude():
    assert tolerance_for(0.8) == pytest.approx(0.016)
    assert tolerance_for(10.0) == pytest.approx(0.2)
    assert tolerance_for(0.001) == pytest.approx(0.005)  # absolute floor


def test_render_table_includes_all_rows():
    rows = (
        ComparisonRow("a", "roc_auc", 0.8, 0.81, 0.01, "PASS"),
        ComparisonRow("b", "rmse", None, 3.2, None, "NEW"),
    )
    table = render_table(rows)
    assert "PASS" in table and "NEW" in table


def test_runner_cli_update_then_detect_regression(tmp_path, monkeypatch):
    """Full CLI path: run_one + main() exit codes, ref writing, drift detection.

    Uses the synthetic benchmark only (no network). Also proves temp-store
    cleanup works on Windows (leaked sqlite handles used to crash rmtree).
    """
    import json

    from benchmarks import datasets, run_benchmarks

    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path / "data")
    refs = tmp_path / "refs.json"
    results = tmp_path / "results.json"
    args = ["--only", "temporal_churn", "--update-refs"]

    # First run: no references yet -> NEW rows never fail; refs get written.
    assert run_benchmarks.main(args, refs_path=refs, results_path=results) == 0
    written = json.loads(refs.read_text(encoding="utf-8"))
    assert written["temporal_churn"]["metric"] == "roc_auc"

    # Sabotage the reference: rerun WITHOUT --update-refs must flag regression.
    written["temporal_churn"]["mean"] = 0.99
    refs.write_text(json.dumps(written), encoding="utf-8")
    rc = run_benchmarks.main(["--only", "temporal_churn"], refs_path=refs, results_path=results)
    assert rc == 1

    # References were NOT silently overwritten by the comparison run.
    assert json.loads(refs.read_text(encoding="utf-8"))["temporal_churn"]["mean"] == 0.99


def test_temporal_dataset_trains_end_to_end(tmp_path):
    """The datetime column must flow through the whole pipeline (epoch cast)."""
    from mlteam.orchestration import run_tabular_pipeline
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore

    data_path = tmp_path / "temporal.parquet"
    make_temporal_churn_frame(n_rows=200).write_parquet(data_path)
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    result = run_tabular_pipeline(
        store,
        data_path,
        target_column="churned",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        seed=42,
        n_folds=3,
        include_tabpfn=False,
        lgbm_config={"n_estimators": 20},
    )
    assert result.evaluation_report.holdout_untouched is True
