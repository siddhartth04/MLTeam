"""Data triage: detecting and fixing trainability blockers before the pipeline runs."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from mlteam.agents.data_io import store_dataframe
from mlteam.agents.triage import triage_dataset
from mlteam.contracts import DatasetRole, TaskType, TriageAction

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
    """Rows with null target are dropped and logged; enough labeled rows remain."""
    n_labeled = 60
    n_missing = 10
    df = pl.DataFrame(
        {
            "feature": list(range(n_labeled + n_missing)),
            "target": [i % 2 for i in range(n_labeled)] + [None] * n_missing,
        }
    )
    ds_id = _store_df(local_store, df, "target")
    triaged, report = triage_dataset(
        local_store,
        ds_id,
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
    )
    assert report.rows_before == n_labeled + n_missing
    assert report.rows_after == n_labeled
    assert len(report.remedies) == 1
    assert report.remedies[0].action == TriageAction.DROP_UNLABELED_ROWS
    assert report.remedies[0].affected_rows == n_missing


def test_triage_coerces_numeric_target_stored_as_text(local_store):
    """Regression target stored as text (e.g. '₹1,000') gets parsed."""
    n = 60
    prices = [f"₹{1000 + i * 100}" if i % 3 == 0 else str(1000 + i * 100) for i in range(n)]
    df = pl.DataFrame({"feature": list(range(n)), "price": prices})
    ds_id = _store_df(local_store, df, "price")
    triaged, report = triage_dataset(
        local_store, ds_id, target_column="price", task_type=TaskType.REGRESSION, run_id=RUN
    )
    assert report.rows_before == n
    assert report.rows_after == n
    assert len(report.remedies) == 1
    assert report.remedies[0].action == TriageAction.COERCE_TARGET_NUMERIC


def test_triage_coerces_junk_numeric_features(local_store):
    """A feature column that's >80% numeric gets junk values nulled."""
    n = 60
    balcony = [str(i % 5) for i in range(n - 1)] + ["> 10"]
    df = pl.DataFrame({"balcony": balcony, "target": [i % 2 for i in range(n)]})
    ds_id = _store_df(local_store, df, "target")
    triaged, report = triage_dataset(
        local_store,
        ds_id,
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
    )
    assert report.rows_before == n
    assert report.rows_after == n
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
            local_store,
            ds_id,
            target_column="target",
            task_type=TaskType.BINARY_CLASSIFICATION,
            run_id=RUN,
        )


def test_triage_preserves_classification_labels_that_look_like_missing(local_store):
    """Classification targets like 'na' or 'none' are NOT dropped — they're class labels."""
    n = 60
    labels = ["yes", "none", "na", "no"] * (n // 4)
    df = pl.DataFrame({"feature": list(range(n)), "class": labels})
    ds_id = _store_df(local_store, df, "class")
    triaged, report = triage_dataset(
        local_store,
        ds_id,
        target_column="class",
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
    )
    assert report.rows_before == n
    assert report.rows_after == n  # no rows dropped
    assert len(report.remedies) == 0  # no changes


def test_kaggle_house_prices_messy_data(local_store, tmp_path: Path):
    """Real-world Kaggle dataset: unlabeled rows, currency-formatted price, > 10 bucket."""
    rows = ["Balcony,Price (in rupees)"]
    rows += [f"{i % 5},₹{100000 + i * 10000}" for i in range(100)]
    # Add missing-price rows
    rows += ["0,"] * 50
    # Add bucket-label rows
    rows.append("> 10,₹5000000")

    path = tmp_path / "house_prices.csv"
    path.write_text("\n".join(rows), encoding="utf-8")

    from mlteam.agents.ingest import ingest_file

    raw = ingest_file(
        local_store,
        path,
        target_column="Price (in rupees)",
        run_id=RUN,
        allow_missing_target=True,
    )
    triaged, report = triage_dataset(
        local_store,
        raw.meta.artifact_id,
        target_column="Price (in rupees)",
        task_type=TaskType.REGRESSION,
        run_id=RUN,
    )

    # Started with 100 + 50 + 1 = 151 rows
    assert report.rows_before == 151
    # 50 rows had no price and are dropped; 101 remain (well above the floor)
    assert report.rows_after >= 50
    assert len(report.remedies) >= 1  # at least one remedy applied
