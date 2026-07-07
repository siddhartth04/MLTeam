"""Out-of-fold predictions: the shared substrate for valid explainability
(audit Task 6) and calibration/cost-optimal thresholds (audit Task 7).

Permutation importance and calibration both require predictions on data the
model did NOT see during fitting — computing either on the training set
with the FINAL model (fit on all training data) rewards memorized features
and produces optimistic calibration, which sklearn's own documentation
warns against. Cross-validated (out-of-fold) predictions give every row a
prediction from a model that never saw it, at the cost of one extra
CV pass reusing the exact same fold assignment training used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mlteam.agents.data_io import read_dataset
from mlteam.agents.trainer import _column_roles, _make_cv, _temporal_to_epoch
from mlteam.contracts import TaskType
from mlteam.store.interface import ArtifactStore
from mlteam.templates.registry import get_template


@dataclass(frozen=True)
class OOFResult:
    """Out-of-fold predictions/probabilities plus per-fold fitted pipelines.

    ``y_pred`` is out-of-fold POINT predictions (regression value, or the
    predicted class for classification) in original row order. ``y_proba``
    is the out-of-fold positive-class probability for binary classification
    (None for regression/multiclass, where a single probability column
    isn't well-defined). ``fold_pipelines`` and ``fold_test_indices`` let a
    caller (permutation importance) recompute per-fold, test-fold-only
    scores without re-fitting anything.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    fold_pipelines: tuple[Any, ...]
    fold_test_indices: tuple[np.ndarray, ...]
    feature_frame: Any  # the pandas DataFrame predictions/importances are indexed against


def compute_oof(
    store: ArtifactStore,
    dataset_id: str,
    *,
    template_id: str,
    config: dict[str, Any],
    task_type: TaskType,
    seed: int,
    n_folds: int,
) -> OOFResult:
    """Refit the template per fold (same CV split training used) and
    collect out-of-fold predictions. This is NOT the trained model's CV
    metrics recomputed — it is genuinely out-of-sample predictions for
    EVERY row, usable for permutation importance, residual analysis, and
    calibration without a second holdout."""
    from sklearn.base import clone

    dataset = store.get(dataset_id)
    if dataset.target_column is None:
        raise ValueError("OOF computation needs a dataset with a declared target column")
    df = _temporal_to_epoch(read_dataset(store, dataset))
    roles = _column_roles(df, dataset.target_column)
    frame = df.to_pandas()
    target = frame.pop(dataset.target_column)

    template = get_template(template_id)
    base_pipeline = template.build(task_type, roles, config, seed)
    cv = _make_cv(task_type, n_folds, seed)

    n = len(frame)
    y_pred = np.empty(n, dtype=object)
    y_proba: np.ndarray | None = None
    has_proba = hasattr(base_pipeline, "predict_proba")
    if has_proba:
        y_proba = np.full(n, np.nan, dtype=np.float64)

    fold_pipelines: list[Any] = []
    fold_test_indices: list[np.ndarray] = []
    for train_idx, test_idx in cv.split(frame, target):
        fitted = clone(base_pipeline).fit(frame.iloc[train_idx], target.iloc[train_idx])
        preds = fitted.predict(frame.iloc[test_idx])
        y_pred[test_idx] = preds
        if has_proba:
            proba = fitted.predict_proba(frame.iloc[test_idx])
            if proba.ndim == 2 and proba.shape[1] == 2:
                y_proba[test_idx] = proba[:, 1]
        fold_pipelines.append(fitted)
        fold_test_indices.append(test_idx)

    return OOFResult(
        y_true=target.to_numpy(),
        y_pred=np.asarray(y_pred.tolist()),
        y_proba=y_proba,
        fold_pipelines=tuple(fold_pipelines),
        fold_test_indices=tuple(fold_test_indices),
        feature_frame=frame,
    )
