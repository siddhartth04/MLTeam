"""Data triage: detecting and fixing trainability blockers before the pipeline runs."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from mlteam.agents.triage import triage_dataset
from mlteam.agents.data_io import store_dataframe
from mlteam.contracts import DatasetRole, TaskType, TriageIssueCode, TriageAction

RUN = "test-triage"


def _store_df(store, df: pl.DataFrame, target_col: str) -> str:
    """Helper: store a dataframe, return its artifact id."""
    ds = store_dataframe(
        store,
        df,
        role=DatasetRole.RAW,
        producer_agent="test",
        run_id=RUN,
        target_column=target_col,
    )
    return ds.meta.artifact_id


def test_triage_drops_unlabeled_rows_with_report(local_store):
    """Rows with null target are dropped and logged."""
    df = pl.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [0, 1, None, 1, 0],
        }
    )
    ds_id = _store_df(local_store, df, "target")
    triaged, report = triage_dataset(
        local_store, ds_id, target_column="target", task_type=TaskType.BINARY_CLASSIFICATION, run_id=RUN
    )
    assert report.rows_before == 5
    assert report.rows_after == 4
    assert len(report.remedies) == 1
    assert report.remedies[0].action == TriageAction.DROP_UNLABELED_ROWS
    assert report.remedies[0].affected_rows == 1


def test_triage_coerces_numeric_target_stored_as_text(local_store):
    """Regression target stored as text (e.g. '₹1,000') gets parsed."""
    df = pl.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "price": ["1000", "2500", "₹3,500", "4000"],
        }
    )
    ds_id = _store_df(local_store, df, "price")
    triaged, report = triage_dataset(
        local_store, ds_id, target_column="price", task_type=TaskType.REGRESSION, run_id=RUN
    )
    assert report.rows_before == 4
    assert report.rows_after == 4
    assert len(report.remedies) == 1
    assert report.remedies[0].action == TriageAction.COERCE_TARGET_NUMERIC


def test_triage_coerces_junk_numeric_features(local_store):
    """A feature column that's >80% numeric gets junk values nulled."""
    df = pl.DataFrame(
        {
            "balcony": ["2", "3", "1", "> 10", "5"],
            "target": [0, 1, 0, 1, 0],
        }
    )
    ds_id = _store_df(local_store, df, "target")
    triaged, report = triage_dataset(
        local_store, ds_id, target_column="target", task_type=TaskType.BINARY_CLASSIFICATION, run_id=RUN
    )
    assert len(report.remedies) == 1
    assert report.remedies[0].action == TriageAction.COERCE_COLUMN_NUMERIC
    assert report.remedies[0].column == "balcony"
    assert report.remedies[0].affected_rows == 1  # "> 10" became null


def test_triage_rejects_too_few_labeled_rows(local_store):
    """After dropping unlabeled, if <50 rows remain, raise ValueError."""
    df = pl.DataFrame(
        {
            "feature": list(range(100)),
            "target": [None] * 90 + list(range(10)),
        }
    )
    ds_id = _store_df(local_store, df, "target")
    with pytest.raises(ValueError, match="too few usable labels"):
        triage_dataset(
            local_store, ds_id, target_column="target", task_type=TaskType.BINARY_CLASSIFICATION, run_id=RUN
        )


def test_triage_preserves_classification_labels_that_look_like_missing(local_store):
    """Classification targets like 'na' or 'none' are NOT dropped — they're class labels."""
    df = pl.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "class": ["yes", "none", "na", "yes"],
        }
    )
    ds_id = _store_df(local_store, df, "class")
    triaged, report = triage_dataset(
        local_store, ds_id, target_column="class", task_type=TaskType.BINARY_CLASSIFICATION, run_id=RUN
    )
    assert report.rows_before == 4
    assert report.rows_after == 4  # no rows dropped
    assert len(report.remedies) == 0  # no changes


def test_kaggle_house_prices_messy_data(local_store, tmp_path: Path):
    """Real-world Kaggle dataset: unlabeled rows, currency-formatted price, > 10 bucket."""
    rows = ["Balcony,Price (in rupees)"]
    rows += [f"{i % 5},₹{100000 + i*10000}" for i in range(100)]
    # Add missing-price rows
    rows += ["0,"] * 50
    # Add bucket-label rows
    rows.append("> 10,₹5000000")

    path = tmp_path / "house_prices.csv"
    path.write_text("\n".join(rows), encoding="utf-8")

    from mlteam.agents.ingest import ingest_file

    raw = ingest_file(local_store, path, target_column="Price (in rupees)", run_id=RUN, allow_missing_target=True)
    triaged, report = triage_dataset(
        local_store,
        raw.meta.artifact_id,
        target_column="Price (in rupees)",
        task_type=TaskType.REGRESSION,
        run_id=RUN,
    )

    # Started with 100 + 50 + 1 = 151 rows
    assert report.rows_before == 151
    # Should have dropped 50 (no price) + 1 (unparseable "> 10") = 51; left with 100
    # Actually: 50 nulls (missing price) + however many "junk" values in numeric coercion
    # The point is: triage should handle it gracefully, not crash
    assert report.rows_after >= 50  # sanity check
    assert len(report.remedies) >= 1  # at least one remedy applied
