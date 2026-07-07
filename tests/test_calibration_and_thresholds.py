"""Audit Task 7 (H7, H8, M6): cost-optimal decision threshold + measured
calibration.

Root causes:
(H8) EvaluationReport.calibration was never populated by any producer; the
     critic's calibration_acceptable gate read expected_calibration_error,
     always got None, and always passed -- a gate that structurally cannot
     fail is audit theater.
(H7) ProblemSpec.cost_asymmetry is solicited from a human at Gate 1 and then
     never read by training/evaluation/reporting -- the platform proceeds
     to optimize threshold-free rank metrics (AUC) and reports accuracy/F1
     at an implicit, unexamined 0.5 cutoff regardless of the stated
     business cost ratio.
(M6) The profiler emits a TARGET_IMBALANCE warning and nothing acts on it;
     the planner's config whitelist had no class_weight lever at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlteam.evaluation.calibration import (
    brier_score,
    compute_calibration,
    expected_calibration_error,
)
from mlteam.evaluation.thresholds import cost_optimal_threshold


def test_perfectly_calibrated_probabilities_have_zero_ece():
    """10 samples predicted p=0.1 with exactly 1 positive (10% actual rate),
    10 samples predicted p=0.9 with exactly 9 positives (90% actual rate) --
    predicted probability matches observed frequency exactly in both bins."""
    y_true = np.array([1] + [0] * 9 + [1] * 9 + [0])
    y_proba = np.array([0.1] * 10 + [0.9] * 10)
    assert expected_calibration_error(y_true, y_proba, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_miscalibrated_probabilities_hand_computed_ece():
    """10 samples all predicted p=0.9, but only 1 is actually positive
    (10% actual rate). Single bin: |0.9 - 0.1| = 0.8 exactly."""
    y_true = np.array([1] + [0] * 9)
    y_proba = np.array([0.9] * 10)
    assert expected_calibration_error(y_true, y_proba, n_bins=10) == pytest.approx(0.8, abs=1e-9)


def test_brier_score_hand_computed():
    """Brier = mean((p - y)^2). p=[0.9]*9 + [0.1], y=[1]*9+[0]:
    9 terms of (0.9-1)^2=0.01, 1 term of (0.1-0)^2=0.01 -> mean = 0.01."""
    y_true = np.array([1] * 9 + [0])
    y_proba = np.array([0.9] * 9 + [0.1])
    assert brier_score(y_true, y_proba) == pytest.approx(0.01, abs=1e-9)


def test_compute_calibration_bundles_both_metrics():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.8, 0.2, 0.8, 0.2])
    result = compute_calibration(y_true, y_proba, n_bins=5)
    assert result.brier_score == pytest.approx(0.04, abs=1e-9)
    assert result.n_bins == 5


def test_cost_optimal_threshold_moves_down_when_false_negatives_costlier():
    """Overlapping distributions (negatives skewed low via Beta(2,5),
    positives skewed higher via Beta(4,3)) so the threshold choice actually
    changes outcomes -- unlike perfectly-separable classes where every
    threshold in the gap achieves zero cost."""
    rng = np.random.default_rng(0)
    n_neg, n_pos = 200, 50
    y_true = np.array([0] * n_neg + [1] * n_pos)
    y_proba = np.concatenate([rng.beta(2, 5, n_neg), rng.beta(4, 3, n_pos)])

    symmetric = cost_optimal_threshold(
        y_true, y_proba, false_positive_cost=1.0, false_negative_cost=1.0
    )
    asymmetric = cost_optimal_threshold(
        y_true, y_proba, false_positive_cost=1.0, false_negative_cost=5.0
    )
    assert asymmetric.threshold < symmetric.threshold
    assert asymmetric.false_negatives <= symmetric.false_negatives
    assert asymmetric.expected_cost <= asymmetric.cost_at_default


def test_cost_optimal_threshold_confusion_counts_are_consistent():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_proba = np.array([0.9, 0.8, 0.3, 0.2, 0.6, 0.7])
    result = cost_optimal_threshold(
        y_true, y_proba, false_positive_cost=1.0, false_negative_cost=1.0
    )
    total = (
        result.true_positives
        + result.false_positives
        + result.false_negatives
        + result.true_negatives
    )
    assert total == len(y_true)
    expected_cost = (
        result.false_positives * result.false_positive_cost
        + result.false_negatives * result.false_negative_cost
    )
    assert result.expected_cost == pytest.approx(expected_cost, abs=1e-6)
