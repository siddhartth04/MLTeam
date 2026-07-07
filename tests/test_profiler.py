"""Profiler tests: stats, warnings, and target relationships."""

from __future__ import annotations

import pytest

from mlteam.agents import ingest_file, profile_dataset
from mlteam.contracts import WarningCode

RUN = "test-profile-run"


@pytest.fixture()
def profile(guarded_store, churn_csv):
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    return profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN, seed=42)


def test_profile_covers_all_columns(profile):
    names = {c.name for c in profile.columns}
    assert {"row_id", "tenure_months", "contract_type", "churned"} <= names


def test_numeric_stats_present(profile):
    tenure = next(c for c in profile.columns if c.name == "tenure_months")
    assert tenure.mean is not None
    assert tenure.std is not None
    assert tenure.missing_pct == 0.0


def test_missingness_recorded(profile):
    charges = next(c for c in profile.columns if c.name == "monthly_charges")
    assert charges.missing_pct > 0.0


def test_constant_column_warning(profile):
    codes = {(w.code, w.column) for w in profile.warnings}
    assert (WarningCode.CONSTANT_COLUMN, "region") in codes


def test_suspected_id_warning(profile):
    codes = {(w.code, w.column) for w in profile.warnings}
    assert any(code is WarningCode.SUSPECTED_ID for code, _ in codes)


def test_duplicate_rows_warning(profile):
    assert any(w.code is WarningCode.DUPLICATE_ROWS for w in profile.warnings)


def test_target_relationships_computed(profile):
    columns = {rel.column for rel in profile.target_relationships}
    assert "tenure_months" in columns
    assert "churned" not in columns


def test_clean_data_has_no_leak_warning(profile):
    assert not any(w.code is WarningCode.SUSPECTED_LEAK for w in profile.warnings)


def test_profile_is_deterministic(guarded_store, churn_csv):
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    a = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id="run-a", seed=42)
    b = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id="run-b", seed=42)
    assert a.meta.artifact_id == b.meta.artifact_id


def test_imbalance_denominator_excludes_unlabeled_rows():
    """audit L5: the minority-class fraction must be computed over LABELED
    rows only. 900 unlabeled + 100 labeled (80/20 split within the labeled
    rows) is NOT imbalanced by any reasonable standard (20% minority), but
    dividing by df.height (1000) would report a fake 8% minority and
    incorrectly trip the warning."""
    import polars as pl

    from mlteam.agents.profiler import _dataset_warnings

    labeled = [1] * 80 + [0] * 20  # 20% minority among labeled rows -- not imbalanced
    unlabeled = [None] * 900
    df = pl.DataFrame({"target": labeled + unlabeled})
    warnings = _dataset_warnings(df, "target", ())
    assert not any(w.code is WarningCode.TARGET_IMBALANCE for w in warnings)


def test_imbalance_denominator_still_flags_real_imbalance_among_labeled_rows():
    import polars as pl

    from mlteam.agents.profiler import _dataset_warnings

    labeled = [1] * 95 + [0] * 5  # 5% minority among labeled rows -- genuinely imbalanced
    unlabeled = [None] * 900
    df = pl.DataFrame({"target": labeled + unlabeled})
    warnings = _dataset_warnings(df, "target", ())
    assert any(w.code is WarningCode.TARGET_IMBALANCE for w in warnings)
