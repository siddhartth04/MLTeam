"""JUDGMENT_ROADMAP.md Task 2: log1p target transformation.

wrap_target_transform's own unit behavior (including its loud failures),
plus the two places a transform can go silently wrong if not threaded
correctly: training on a target the transform can't handle (negative
values), and out-of-fold predictions landing on the wrong scale (log,
not original) and quietly corrupting permutation importance / residuals.
The end-to-end pipeline behavior (winner picks it up automatically, beats
a raw-fit comparison) is asserted in tests/test_judgment_traps.py's
skewed_price_target trap -- that trap's flip IS this task's acceptance test.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlteam.agents.data_io import store_dataframe
from mlteam.agents.trainer import compute_env_hash, train_experiment
from mlteam.contracts import BudgetEstimate, DatasetRole, ExperimentSpec, TaskType
from mlteam.evaluation.oof import compute_oof
from mlteam.templates.registry import wrap_target_transform

RUN = "test-target-transform"


def _store_df(store, df: pl.DataFrame, target_col: str) -> str:
    ds = store_dataframe(
        store,
        df,
        role=DatasetRole.TRAIN,
        producer_agent="test",
        run_id=RUN,
        target_column=target_col,
    )
    return ds.meta.artifact_id


def test_wrap_target_transform_returns_pipeline_unchanged_for_none():
    pipeline = object()
    assert wrap_target_transform(pipeline, None, TaskType.REGRESSION) is pipeline


def test_wrap_target_transform_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown target_transform"):
        wrap_target_transform(object(), "sqrt", TaskType.REGRESSION)


def test_wrap_target_transform_rejects_non_regression_task():
    with pytest.raises(ValueError, match="only supported for"):
        wrap_target_transform(object(), "log1p", TaskType.BINARY_CLASSIFICATION)


def test_wrap_target_transform_wraps_log1p_in_transformed_target_regressor():
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import LinearRegression

    wrapped = wrap_target_transform(LinearRegression(), "log1p", TaskType.REGRESSION)
    assert isinstance(wrapped, TransformedTargetRegressor)
    assert wrapped.func is np.log1p
    assert wrapped.inverse_func is np.expm1


def test_train_experiment_rejects_negative_target_with_log1p(guarded_store):
    """The fit itself must refuse, with a comprehensible message -- log1p
    of a value below -1 is undefined; between -1 and 0 it silently produces
    a valid-looking but meaningless output, so this must be loud, not clipped."""
    df = pl.DataFrame(
        {
            "feature": list(range(60)),
            "price": [-5.0] + [float(i) for i in range(59)],
        }
    )
    ds_id = _store_df(guarded_store, df, "price")
    spec = ExperimentSpec(
        experiment_id="exp-log1p-negative",
        template_id="lightgbm_default",
        budget=BudgetEstimate(wall_time_seconds=5),
        is_baseline=False,
        target_transform="log1p",
    )
    with pytest.raises(ValueError, match="non-negative target"):
        train_experiment(
            guarded_store,
            ds_id,
            spec,
            task_type=TaskType.REGRESSION,
            run_id=RUN,
            seed=42,
            n_folds=3,
            env_hash=compute_env_hash(),
        )


def test_compute_oof_with_log1p_produces_original_scale_predictions(guarded_store):
    """OOF predictions feed permutation importance and residual slicing --
    if they land on log-scale (~0-3 here) instead of original units
    (~10-150), both go silently wrong without ever raising."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.uniform(1, 10, n)
    price = np.expm1(2.0 + 0.3 * x + rng.normal(0, 0.2, n))
    df = pl.DataFrame({"feature": x, "price": price})
    ds_id = _store_df(guarded_store, df, "price")

    oof = compute_oof(
        guarded_store,
        ds_id,
        template_id="lightgbm_default",
        config={"n_estimators": 30},
        task_type=TaskType.REGRESSION,
        seed=42,
        n_folds=3,
        target_transform="log1p",
    )
    assert np.median(oof.y_pred) > 10  # original units; log-scale would be ~0-3
