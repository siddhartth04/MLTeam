"""Ingestion robustness: real-world CSVs with mixed-type columns."""

from __future__ import annotations

from pathlib import Path

from mlteam.agents.ingest import ingest_file

RUN = "test-ingest-run"


def test_mixed_type_bucket_column_does_not_crash_ingestion(local_store, tmp_path: Path):
    """A numeric column with a '> 10' bucket label (common on Kaggle datasets)
    must ingest, not raise polars.exceptions.ComputeError."""
    rows = ["balcony,price,label"]
    rows += [f"{i % 5},{100 + i},{i % 2}" for i in range(60)]
    rows.append("> 10,999,1")  # the bucket-label row that broke strict inference
    path = tmp_path / "listings.csv"
    path.write_text("\n".join(rows), encoding="utf-8")

    dataset = ingest_file(local_store, path, target_column="label", run_id=RUN)
    assert dataset.n_rows == 61


def test_ordinary_csv_still_ingests_cleanly(local_store, tmp_path: Path):
    path = tmp_path / "plain.csv"
    path.write_text("a,b,target\n1,2,0\n3,4,1\n", encoding="utf-8")
    dataset = ingest_file(local_store, path, target_column="target", run_id=RUN)
    assert dataset.n_rows == 2


def test_read_columns_ignores_mixed_type_values():
    """_read_columns only needs names — must not choke on bucket-label rows
    beyond its old 50-row inference window."""
    from mlteam.orchestration.agentic import _read_columns

    rows = ["a,b"] + [f"{i},{i}" for i in range(60)] + ["> 10,1"]
    path = Path("_scratch_read_columns_test.csv")
    try:
        path.write_text("\n".join(rows), encoding="utf-8")
        assert _read_columns(path) == ["a", "b"]
    finally:
        path.unlink(missing_ok=True)
