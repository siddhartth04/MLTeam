"""Deep stacking, seasonal naive floor, and quantile forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlteam.contracts import TaskType
from mlteam.templates.registry import ColumnRoles, get_template

SEED = 21
ROLES = ColumnRoles(numeric=("x1", "x2"), categorical=())


def _clf_data(n: int = 90):
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = pd.Series(np.where(X["x1"] + rng.normal(scale=0.5, size=n) > 0, "yes", "no"))
    return X, y


def _reg_data(n: int = 90):
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = pd.Series(2.0 * X["x1"] - X["x2"] + rng.normal(scale=0.3, size=n))
    return X, y


def test_deep_stack_classification():
    X, y = _clf_data()
    pipeline = get_template("stacking_deep_default").build(
        TaskType.BINARY_CLASSIFICATION, ROLES, {}, SEED
    )
    pipeline.fit(X, y)
    assert set(pipeline.predict(X)) <= {"yes", "no"}
    assert pipeline.predict_proba(X).shape == (len(X), 2)


def test_deep_stack_regression():
    X, y = _reg_data()
    pipeline = get_template("stacking_deep_default").build(TaskType.REGRESSION, ROLES, {}, SEED)
    pipeline.fit(X, y)
    assert np.isfinite(pipeline.predict(X)).all()


def test_seasonal_naive_predicts_lag_column():
    n = 40
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame(
        {
            "promo": rng.integers(0, 2, size=n),
            "sales_lag_1": rng.uniform(50, 100, size=n),
            "sales_lag_7": rng.uniform(50, 100, size=n),
        }
    )
    y = pd.Series(rng.uniform(50, 100, size=n))
    model = get_template("seasonal_naive_default").build(TaskType.FORECASTING, ROLES, {}, SEED)
    model.fit(X, y)
    np.testing.assert_allclose(model.predict(X), X["sales_lag_7"].to_numpy())


def test_seasonal_naive_requires_lag_column():
    X = pd.DataFrame({"promo": [0, 1, 0]})
    y = pd.Series([1.0, 2.0, 3.0])
    model = get_template("seasonal_naive_default").build(TaskType.FORECASTING, ROLES, {}, SEED)
    with pytest.raises(ValueError, match="lag column"):
        model.fit(X, y)


def test_seasonal_naive_survives_cloning():
    from sklearn.base import clone

    model = get_template("seasonal_naive_default").build(
        TaskType.FORECASTING, ROLES, {"season_length": 7}, SEED
    )
    cloned = clone(model)
    assert cloned.get_params()["season_length"] == 7


def test_quantile_gbm_median_and_intervals():
    X, y = _reg_data(n=150)
    pipeline = get_template("lightgbm_quantile_default").build(
        TaskType.FORECASTING, ROLES, {"n_estimators": 60}, SEED
    )
    pipeline.fit(X, y)
    median = pipeline.predict(X)
    assert np.isfinite(median).all()
    bands = pipeline.named_steps["model"].predict_quantiles(
        pipeline.named_steps["preprocess"].transform(X)
    )
    assert bands.shape == (len(X), 3)
    # audit L1: predict_quantiles sorts per-row, so non-crossing is now
    # guaranteed for every row, not just "most" of them (three independently
    # trained quantile models have no structural reason to agree on order).
    ordered = (bands[:, 0] <= bands[:, 1] + 1e-9) & (bands[:, 1] <= bands[:, 2] + 1e-9)
    assert ordered.all()


def test_quantile_gbm_sorts_crossing_quantiles_at_predict():
    """audit L1: force a crossing (lower quantile model predicts ABOVE the
    upper quantile model for some rows, which independently trained boosted
    models can do on real data) and confirm predict_quantiles repairs it by
    sorting rather than reporting an inverted interval."""
    from unittest.mock import MagicMock

    import numpy as np

    from mlteam.templates.registry import QuantileGBMRegressor

    reg = QuantileGBMRegressor(quantiles=(0.05, 0.5, 0.95))
    # Fake fitted sub-models whose raw predictions deliberately cross.
    reg.models_ = {
        0.05: MagicMock(predict=lambda X: np.array([10.0, 1.0])),  # crosses row 1
        0.5: MagicMock(predict=lambda X: np.array([5.0, 5.0])),
        0.95: MagicMock(predict=lambda X: np.array([8.0, 9.0])),
    }
    bands = reg.predict_quantiles(np.zeros((2, 1)))
    assert (bands[:, 0] <= bands[:, 1]).all()
    assert (bands[:, 1] <= bands[:, 2]).all()
    # row 0's raw values (10, 5, 8) sorted ascending -> (5, 8, 10)
    assert list(bands[0]) == [5.0, 8.0, 10.0]


def test_quantile_gbm_survives_cv():
    from sklearn.model_selection import cross_val_score

    X, y = _reg_data(n=120)
    pipeline = get_template("lightgbm_quantile_default").build(
        TaskType.REGRESSION, ROLES, {"n_estimators": 40}, SEED
    )
    scores = cross_val_score(pipeline, X, y, cv=3, scoring="neg_root_mean_squared_error")
    assert len(scores) == 3


def test_forecasting_registry_lineup():
    from mlteam.templates.registry import list_templates

    ids = {t.template_id for t in list_templates(TaskType.FORECASTING)}
    assert {"seasonal_naive_default", "lightgbm_quantile_default", "stacking_deep_default"} <= ids
    # seasonal naive must not leak into plain regression
    reg_ids = {t.template_id for t in list_templates(TaskType.REGRESSION)}
    assert "seasonal_naive_default" not in reg_ids
    assert "lightgbm_quantile_default" in reg_ids
