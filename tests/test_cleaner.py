"""Cleaner tests: rule-based proposals, declarative execution, immutability."""

from __future__ import annotations

import pytest

from mlteam.agents import (
    execute_cleaning,
    ingest_file,
    profile_dataset,
    propose_cleaning_ops,
    read_dataset,
)
from mlteam.contracts import CleaningOpType

RUN = "test-clean-run"


@pytest.fixture()
def setup(guarded_store, churn_csv):
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    profile = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)
    return dataset, profile


def test_proposals_target_known_problems(setup):
    _, profile = setup
    ops = propose_cleaning_ops(profile)
    op_types = {op.op_type for op in ops}
    assert CleaningOpType.DROP_COLUMN in op_types  # constant + id columns
    assert CleaningOpType.DROP_DUPLICATE_ROWS in op_types
    assert CleaningOpType.STANDARDIZE_MISSING_TOKENS in op_types
    assert all(op.justification for op in ops)


def test_execution_produces_new_dataset(guarded_store, setup):
    dataset, profile = setup
    cleaned, manifest = execute_cleaning(
        guarded_store, dataset.meta.artifact_id, propose_cleaning_ops(profile), run_id=RUN
    )
    assert cleaned.meta.artifact_id != dataset.meta.artifact_id
    assert manifest.input_dataset_ref.artifact_id == dataset.meta.artifact_id
    assert manifest.output_dataset_ref.artifact_id == cleaned.meta.artifact_id

    df = read_dataset(guarded_store, cleaned)
    assert "region" not in df.columns  # constant dropped
    assert "row_id" not in df.columns  # id dropped
    assert df.height < dataset.n_rows  # duplicates removed


def test_source_dataset_is_untouched(guarded_store, setup):
    dataset, profile = setup
    before = read_dataset(guarded_store, dataset)
    execute_cleaning(
        guarded_store, dataset.meta.artifact_id, propose_cleaning_ops(profile), run_id=RUN
    )
    after = read_dataset(guarded_store, dataset)
    assert before.equals(after)


def test_missing_tokens_become_nulls(guarded_store, setup):
    dataset, profile = setup
    cleaned, _ = execute_cleaning(
        guarded_store, dataset.meta.artifact_id, propose_cleaning_ops(profile), run_id=RUN
    )
    df = read_dataset(guarded_store, cleaned)
    assert df["contract_type"].null_count() > 0
    assert not df["contract_type"].str.to_lowercase().is_in(["n/a"]).any()


def test_manifest_records_affected_rows(guarded_store, setup):
    dataset, profile = setup
    _, manifest = execute_cleaning(
        guarded_store, dataset.meta.artifact_id, propose_cleaning_ops(profile), run_id=RUN
    )
    dedup = next(op for op in manifest.ops if op.op_type is CleaningOpType.DROP_DUPLICATE_ROWS)
    assert dedup.affected_rows == 3  # conftest appends 3 duplicate rows
