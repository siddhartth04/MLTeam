"""Cost-optimal decision threshold for binary classification.

ProblemSpec.cost_asymmetry captures the business's stated relative cost of
a false positive vs a false negative (e.g. "a missed fraud case costs 5x a
false alarm") — a human sets this at Gate 1 (audit finding H7: previously
solicited and then never read again anywhere). The rank metrics this
platform reports (ROC-AUC, PR-AUC) are threshold-free by design and cannot
express that asymmetry; only a chosen decision threshold can. This module
finds the threshold that minimizes expected cost over out-of-fold
predictions, so the trade-off the human actually specified is the one the
report acts on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Candidate thresholds are swept at this resolution; fine enough that the
# chosen point is never far from the analytic optimum for a smooth cost
# curve, coarse enough to be instant even on tens of thousands of rows.
_THRESHOLD_STEPS = 200


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    expected_cost: float
    cost_at_default: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    false_positive_cost: float
    false_negative_cost: float


def _confusion_counts(y_true: np.ndarray, y_pred_positive: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_pred_positive == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred_positive == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred_positive == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred_positive == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def _expected_cost(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float, *, fp_cost: float, fn_cost: float
) -> tuple[float, int, int, int, int]:
    y_pred_positive = (y_proba >= threshold).astype(int)
    tp, fp, fn, tn = _confusion_counts(y_true, y_pred_positive)
    cost = fp * fp_cost + fn * fn_cost
    return cost, fp, fn, tp, tn


def cost_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> ThresholdResult:
    """Sweep candidate thresholds over OOF probabilities and return the one
    minimizing ``FP * false_positive_cost + FN * false_negative_cost``.

    Ties (multiple thresholds achieving the same minimum cost) are broken by
    preferring the threshold closest to 0.5 — the least aggressive of the
    equally-good options, avoiding an arbitrarily extreme cutoff when the
    cost surface is flat.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    candidates = np.linspace(0.0, 1.0, _THRESHOLD_STEPS + 1)
    best_threshold = 0.5
    best_cost = float("inf")
    best_stats: tuple[int, int, int, int] = (0, 0, 0, 0)
    for t in candidates:
        cost, fp, fn, tp, tn = _expected_cost(
            y_true, y_proba, t, fp_cost=false_positive_cost, fn_cost=false_negative_cost
        )
        if cost < best_cost - 1e-12 or (
            abs(cost - best_cost) <= 1e-12 and abs(t - 0.5) < abs(best_threshold - 0.5)
        ):
            best_cost = cost
            best_threshold = float(t)
            best_stats = (tp, fp, fn, tn)

    default_cost, _, _, _, _ = _expected_cost(
        y_true, y_proba, 0.5, fp_cost=false_positive_cost, fn_cost=false_negative_cost
    )
    tp, fp, fn, tn = best_stats
    return ThresholdResult(
        threshold=round(best_threshold, 4),
        expected_cost=round(best_cost, 6),
        cost_at_default=round(default_cost, 6),
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        true_negatives=tn,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
