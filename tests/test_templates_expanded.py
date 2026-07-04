"""Smoke tests for the expanded template registry: every algorithm family
builds, fits, and predicts on small data for every task type it declares."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlteam.contracts import TaskType, is_classification
from mlteam.evaluation.metrics import scoring_for
from mlteam.templates.registry import ColumnRoles, get_template, list_templates

SEED = 7
N = 120

NEW_TEMPLATES = [
    "xgboost_default",
    "catboost_default",
    "random_forest_default",
    "extra_trees_default",
    "adaboost_default",
    "linear_default",
    "svm_default",
    "knn_default",
    "naive_bayes_default",
    "mlp_default",
]


def _frame(seed: int = SEED) -> tuple[pd.DataFrame, ColumnRoles]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=N),
            "x2": rng.normal(size=N),
            "cat": rng.choice(["a", "b", "c"], size=N),
        }
    )
    roles = ColumnRoles(numeric=("x1", "x2"), categorical=("cat",))
    return frame, roles


def _binary_target(frame: pd.DataFrame, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=0.5, size=len(frame))
    # String labels on purpose: exercises ordinal encoding + the XGBoost
    # label-encoding wrapper.
    return pd.Series(np.where(frame["x1"] + noise > 0, "yes", "no"))


def _regression_target(frame: pd.DataFrame, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(2.0 * frame["x1"] - frame["x2"] + rng.normal(scale=0.3, size=len(frame)))


def test_registry_has_all_new_templates():
    ids = {t.template_id for t in list_templates()}
    for template_id in NEW_TEMPLATES:
        assert template_id in ids


@pytest.mark.parametrize("template_id", NEW_TEMPLATES)
def test_classification_fit_predict(template_id: str):
    template = get_template(template_id)
    if not template.supports(TaskType.BINARY_CLASSIFICATION):
        pytest.skip(f"{template_id} does not support classification")
    frame, roles = _frame()
    y = _binary_target(frame)
    pipeline = template.build(TaskType.BINARY_CLASSIFICATION, roles, {}, SEED)
    pipeline.fit(frame, y)
    preds = pipeline.predict(frame)
    assert set(preds) <= {"yes", "no"}
    proba = pipeline.predict_proba(frame)
    assert proba.shape == (N, 2)


@pytest.mark.parametrize("template_id", NEW_TEMPLATES)
def test_regression_fit_predict(template_id: str):
    template = get_template(template_id)
    if not template.supports(TaskType.REGRESSION):
        pytest.skip(f"{template_id} is classification-only")
    frame, roles = _frame()
    y = _regression_target(frame)
    pipeline = template.build(TaskType.REGRESSION, roles, {}, SEED)
    pipeline.fit(frame, y)
    preds = pipeline.predict(frame)
    assert preds.shape == (N,)
    assert np.isfinite(preds).all()


def test_naive_bayes_is_classification_only():
    template = get_template("naive_bayes_default")
    assert template.supports(TaskType.BINARY_CLASSIFICATION)
    assert not template.supports(TaskType.REGRESSION)
    assert not template.supports(TaskType.FORECASTING)


def test_xgboost_survives_cloning_with_string_labels():
    """cross_validate clones pipelines; the label wrapper must clone cleanly."""
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    frame, roles = _frame()
    y = _binary_target(frame)
    pipeline = get_template("xgboost_default").build(
        TaskType.BINARY_CLASSIFICATION, roles, {"n_estimators": 20}, SEED
    )
    cloned = clone(pipeline)
    cloned.fit(frame, y)
    assert set(cloned.predict(frame)) <= {"yes", "no"}
    scores = cross_val_score(pipeline, frame, y, cv=3, scoring="roc_auc")
    assert len(scores) == 3


def test_forecasting_task_type_wiring():
    assert not is_classification(TaskType.FORECASTING)
    assert not is_classification(TaskType.REGRESSION)
    assert is_classification(TaskType.BINARY_CLASSIFICATION)
    assert scoring_for(TaskType.FORECASTING) == scoring_for(TaskType.REGRESSION)


def test_templates_support_forecasting():
    forecasting_ids = {t.template_id for t in list_templates(TaskType.FORECASTING)}
    assert "lightgbm_default" in forecasting_ids
    assert "xgboost_default" in forecasting_ids
    assert "naive_bayes_default" not in forecasting_ids
