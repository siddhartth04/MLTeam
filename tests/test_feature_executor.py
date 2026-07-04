"""Feature-plan executor: row-local ops apply, stat-learning ops drop loudly."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from mlteam.agents.data_io import store_dataframe
from mlteam.agents.feature_executor import execute_feature_plan
from mlteam.contracts import (
    DatasetRole,
    FeatureOpType,
    FeaturePlan,
    FeatureProposal,
    new_artifact,
)

RUN = "test-feat-exec"


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "income": [50000.0, 0.0, 31000.0, 64000.0],
            "debt": [12000.0, 0.0, 8000.0, 20000.0],
            "signup": [date(2025, 1, 5), date(2025, 3, 2), date(2025, 6, 9), date(2025, 9, 1)],
            "notes": ["called twice", "", "vip customer", "n/a"],
            "segment": ["a", "b", "a", "c"],
            "churned": [0, 1, 0, 1],
        }
    )


@pytest.fixture()
def dataset(local_store):
    return store_dataframe(
        local_store,
        _frame(),
        role=DatasetRole.CLEANED,
        producer_agent="test",
        run_id=RUN,
        target_column="churned",
    )


def _plan(dataset, proposals) -> FeaturePlan:
    return new_artifact(
        FeaturePlan,
        producer_agent="feature-engineer",
        run_id=RUN,
        dataset_ref=dataset.ref(),
        proposals=tuple(proposals),
    )


def _proposal(name, op, cols, params=None):
    return FeatureProposal(
        name=name,
        op_type=op,
        input_columns=tuple(cols),
        params=params or {},
        rationale="test rationale",
    )


def test_row_local_ops_execute(local_store, dataset):
    plan = _plan(
        dataset,
        [
            _proposal("debt_to_income", FeatureOpType.RATIO, ["debt", "income"]),
            _proposal("disposable", FeatureOpType.DIFFERENCE, ["income", "debt"]),
            _proposal("log_income", FeatureOpType.LOG_TRANSFORM, ["income"]),
            _proposal("signup_parts", FeatureOpType.DATETIME_DECOMPOSE, ["signup"]),
            _proposal("note_len", FeatureOpType.TEXT_LENGTH, ["notes"]),
            _proposal(
                "income_band",
                FeatureOpType.BINNING,
                ["income"],
                {"edges": [40000, 70000]},
            ),
        ],
    )
    local_store.put(plan)
    featured, feature_set = execute_feature_plan(
        local_store, dataset.meta.artifact_id, plan, run_id=RUN
    )
    assert featured is not None
    assert feature_set.dropped_proposals == ()
    from mlteam.agents.data_io import read_dataset

    df = read_dataset(local_store, featured)
    for col in (
        "debt_to_income",
        "disposable",
        "log_income",
        "signup_parts_month",
        "signup_parts_dayofweek",
        "note_len",
        "income_band",
    ):
        assert col in df.columns, f"missing {col}"
    # zero denominator -> null, not inf/crash
    assert df["debt_to_income"][1] is None
    assert df["disposable"][0] == 38000.0
    # lineage: featured dataset is a child of cleaned + plan
    assert dataset.meta.artifact_id in featured.meta.parent_ids


def test_stat_learning_ops_drop_loudly(local_store, dataset):
    plan = _plan(
        dataset,
        [
            _proposal("seg_target", FeatureOpType.TARGET_ENCODE, ["segment"]),
            _proposal("seg_freq", FeatureOpType.FREQUENCY_ENCODE, ["segment"]),
            _proposal("seg_group", FeatureOpType.GROUP_AGGREGATE, ["segment", "income"]),
            _proposal("seg_onehot", FeatureOpType.ONE_HOT_ENCODE, ["segment"]),
            _proposal("bad_bins", FeatureOpType.BINNING, ["income"]),  # no edges
            # malformed LLM params must DROP, never crash the run
            _proposal(
                "junk_edges",
                FeatureOpType.BINNING,
                ["income"],
                {"edges": ["low", None, "high"]},
            ),
        ],
    )
    local_store.put(plan)
    featured, feature_set = execute_feature_plan(
        local_store, dataset.meta.artifact_id, plan, run_id=RUN
    )
    assert featured is None  # nothing applicable
    assert set(feature_set.dropped_proposals) == {
        "seg_target",
        "seg_freq",
        "seg_group",
        "seg_onehot",
        "bad_bins",
        "junk_edges",
    }
    assert feature_set.feature_names == ()


def test_name_collision_and_bad_columns_drop(local_store, dataset):
    plan = _plan(
        dataset,
        [
            _proposal("income", FeatureOpType.LOG_TRANSFORM, ["income"]),  # collides
            _proposal("ghost", FeatureOpType.RATIO, ["nope", "income"]),  # missing col
            _proposal("text_ratio", FeatureOpType.RATIO, ["notes", "income"]),  # non-numeric
            _proposal("ok_feature", FeatureOpType.PRODUCT, ["income", "debt"]),
        ],
    )
    local_store.put(plan)
    featured, feature_set = execute_feature_plan(
        local_store, dataset.meta.artifact_id, plan, run_id=RUN
    )
    assert featured is not None
    assert set(feature_set.dropped_proposals) == {"income", "ghost", "text_ratio"}
    assert feature_set.feature_names == ("ok_feature",)
