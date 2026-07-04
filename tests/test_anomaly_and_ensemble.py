"""Anomaly detection task + dedicated detectors + the stacking ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlteam.contracts import TaskType, is_classification
from mlteam.evaluation.metrics import scoring_for
from mlteam.templates.registry import ColumnRoles, get_template, list_templates

SEED = 13
DETECTORS = [
    "isolation_forest_default",
    "one_class_svm_default",
    "local_outlier_factor_default",
]


def _anomaly_data(n: int = 200, contamination: float = 0.1, seed: int = SEED):
    """Two-feature gaussian blob with planted far-away outliers, labels 0/1."""
    rng = np.random.default_rng(seed)
    n_anom = int(n * contamination)
    n_norm = n - n_anom
    normal = rng.normal(0.0, 1.0, size=(n_norm, 2))
    anomalies = rng.normal(0.0, 1.0, size=(n_anom, 2)) + rng.choice(
        [-6.0, 6.0], size=(n_anom, 2)
    )
    X = pd.DataFrame(np.vstack([normal, anomalies]), columns=["x1", "x2"])
    y = pd.Series(np.array([0] * n_norm + [1] * n_anom))
    shuffle = rng.permutation(n)
    return X.iloc[shuffle].reset_index(drop=True), y.iloc[shuffle].reset_index(drop=True)


ROLES = ColumnRoles(numeric=("x1", "x2"), categorical=())


def test_anomaly_task_wiring():
    assert is_classification(TaskType.ANOMALY_DETECTION)
    scoring = scoring_for(TaskType.ANOMALY_DETECTION)
    assert scoring["pr_auc"] == "average_precision"
    assert "roc_auc" in scoring


def test_binary_classification_gained_pr_auc():
    assert scoring_for(TaskType.BINARY_CLASSIFICATION)["pr_auc"] == "average_precision"


def test_detectors_registered_for_anomaly_only():
    anomaly_ids = {t.template_id for t in list_templates(TaskType.ANOMALY_DETECTION)}
    binary_ids = {t.template_id for t in list_templates(TaskType.BINARY_CLASSIFICATION)}
    for det in DETECTORS:
        assert det in anomaly_ids
        assert det not in binary_ids
    # ordinary classifiers compete on anomaly tasks too
    assert "lightgbm_default" in anomaly_ids
    assert "stacking_ensemble_default" in anomaly_ids


@pytest.mark.parametrize("template_id", DETECTORS)
def test_detector_ranks_planted_outliers(template_id: str):
    from sklearn.metrics import roc_auc_score

    X, y = _anomaly_data()
    pipeline = get_template(template_id).build(TaskType.ANOMALY_DETECTION, ROLES, {}, SEED)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    # classes_ [0, 1] -> column 1 is the anomaly probability
    assert roc_auc_score(y, proba[:, 1]) > 0.9
    preds = pipeline.predict(X)
    assert set(np.unique(preds)) <= {0, 1}


def test_detector_survives_cv_cloning():
    from sklearn.model_selection import cross_val_score

    X, y = _anomaly_data()
    pipeline = get_template("isolation_forest_default").build(
        TaskType.ANOMALY_DETECTION, ROLES, {}, SEED
    )
    scores = cross_val_score(pipeline, X, y, cv=3, scoring="average_precision")
    assert len(scores) == 3
    assert scores.mean() > 0.5  # planted far outliers are easy


def test_detector_rejects_multiclass_labels():
    X, y = _anomaly_data()
    y3 = y.copy()
    y3.iloc[:5] = 2
    pipeline = get_template("isolation_forest_default").build(
        TaskType.ANOMALY_DETECTION, ROLES, {}, SEED
    )
    with pytest.raises(ValueError, match="exactly 2 label values"):
        pipeline.fit(X, y3)


def test_stacking_classification_with_string_labels():
    rng = np.random.default_rng(SEED)
    n = 150
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = pd.Series(np.where(X["x1"] + rng.normal(scale=0.5, size=n) > 0, "yes", "no"))
    pipeline = get_template("stacking_ensemble_default").build(
        TaskType.BINARY_CLASSIFICATION, ColumnRoles(("x1", "x2"), ()), {}, SEED
    )
    pipeline.fit(X, y)
    assert set(pipeline.predict(X)) <= {"yes", "no"}
    assert pipeline.predict_proba(X).shape == (n, 2)


def test_stacking_regression():
    rng = np.random.default_rng(SEED)
    n = 150
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = pd.Series(3.0 * X["x1"] - X["x2"] + rng.normal(scale=0.2, size=n))
    pipeline = get_template("stacking_ensemble_default").build(
        TaskType.REGRESSION, ColumnRoles(("x1", "x2"), ()), {}, SEED
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert np.isfinite(preds).all()


def test_anomaly_pipeline_end_to_end(guarded_store, tmp_path: Path):
    import polars as pl

    from mlteam.orchestration.pipeline import run_tabular_pipeline

    X, y = _anomaly_data(n=180)
    frame = pl.DataFrame({"x1": X["x1"], "x2": X["x2"], "is_anomaly": y})
    csv_path = tmp_path / "sensors.csv"
    frame.write_csv(csv_path)

    result = run_tabular_pipeline(
        guarded_store,
        csv_path,
        target_column="is_anomaly",
        task_type=TaskType.ANOMALY_DETECTION,
        primary_metric="pr_auc",
        seed=42,
        n_folds=3,
        include_tabpfn=False,
    )
    report = result.evaluation_report
    metric_names = {m.name for m in report.cv_metrics}
    assert "pr_auc" in metric_names
    split = guarded_store.get(result.split_plan_id)
    assert split.strategy.value == "stratified"
