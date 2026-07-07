"""Calibration metrics: Brier score + binned expected calibration error (ECE).

Both are computed from OUT-OF-FOLD probabilities (evaluation/oof.py) —
calibration measured on the training set with the final model is optimistic
for the same reason permutation importance is (audit Task 6): the model was
fit to (indirectly) match its own training labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_DEFAULT_N_BINS = 10


@dataclass(frozen=True)
class CalibrationResult:
    brier_score: float
    expected_calibration_error: float
    n_bins: int


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean squared error between predicted probability and the binary
    outcome — 0 is perfect, 0.25 is what an uninformative p=0.5 achieves on
    a balanced set."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    return float(np.mean((y_proba - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, *, n_bins: int = _DEFAULT_N_BINS
) -> float:
    """Binned ECE: partition [0,1] into n_bins equal-width buckets by
    predicted probability; within each non-empty bucket compare the mean
    predicted probability to the actual observed positive rate; ECE is the
    bucket-size-weighted average absolute gap. A well-calibrated model (a
    predicted 0.7 really does mean "70% of the time this class occurs") has
    ECE near 0; a model whose confidence doesn't track reality has a large
    gap regardless of how good its RANKING metrics (AUC) look — ranking and
    calibration are different properties a model can have independently.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Rightmost edge inclusive, so a probability of exactly 1.0 lands in the
    # last bin instead of falling out of range.
    bin_idx = np.clip(np.digitize(y_proba, bin_edges[1:-1], right=True), 0, n_bins - 1)

    total_gap = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(y_proba[mask].mean())
        mean_actual = float(y_true[mask].mean())
        total_gap += (count / n) * abs(mean_pred - mean_actual)
    return total_gap


def compute_calibration(
    y_true: np.ndarray, y_proba: np.ndarray, *, n_bins: int = _DEFAULT_N_BINS
) -> CalibrationResult:
    return CalibrationResult(
        brier_score=round(brier_score(y_true, y_proba), 6),
        expected_calibration_error=round(
            expected_calibration_error(y_true, y_proba, n_bins=n_bins), 6
        ),
        n_bins=n_bins,
    )
