"""Paired, corrected significance testing over cross-validation folds.

Two models compared on the SAME k-fold split are not independent samples —
they are paired observations (each fold contributes one comparison), and the
folds themselves are not independent either (adjacent folds share most of
their training data, so fold-to-fold scores are positively correlated). A
plain paired t-test ignores that correlation and is anti-conservative
(overstates significance). The standard correction is the Nadeau-Bengio
resampled t-test (Nadeau & Bengio, 2003, "Inference for the Generalization
Error"; corrected variance estimator per Bouckaert & Frank, 2004): inflate
the naive paired-variance estimate by the train/test size ratio to account
for the overlap, before forming the t-statistic.

This module is pure math — no store, no I/O — so it is trivially unit
testable against hand-computed cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedTestResult:
    """Result of a paired Nadeau-Bengio corrected t-test over fold deltas.

    ``mean_delta`` and ``std_delta`` describe the per-fold difference
    (better - worse in the metric's natural improve-is-positive direction,
    i.e. already sign-normalized by the caller via ``signed_improvement``).
    """

    k: int
    mean_delta: float
    std_delta: float
    t_statistic: float | None
    p_value: float | None
    degrees_of_freedom: int


def paired_fold_deltas(a_folds: tuple[float, ...], b_folds: tuple[float, ...]) -> tuple[float, ...]:
    """Per-fold (a - b), pairing fold i of a with fold i of b.

    Both sequences must come from cross-validation over the SAME fold
    assignment (identical seed, n_folds, and split strategy) for the pairing
    to be meaningful -- callers are responsible for checking that precondition
    (see ``fold_pairing_is_valid``).
    """
    if len(a_folds) != len(b_folds):
        raise ValueError(
            f"cannot pair {len(a_folds)} folds with {len(b_folds)} folds — "
            "fold counts must match for a paired comparison"
        )
    return tuple(a - b for a, b in zip(a_folds, b_folds, strict=True))


def fold_pairing_is_valid(a_folds: tuple[float, ...], b_folds: tuple[float, ...]) -> bool:
    """Whether a paired test can be run at all: at least 2 folds each side,
    same count. This does NOT (cannot, from values alone) verify the fold
    ASSIGNMENTS were identical — callers must ensure that by construction
    (same seed/n_folds/strategy for both experiments being compared)."""
    return len(a_folds) == len(b_folds) and len(a_folds) >= 2


def nadeau_bengio_test(
    deltas: tuple[float, ...],
    *,
    n_train: int,
    n_test: int,
) -> PairedTestResult:
    """Nadeau-Bengio corrected resampled paired t-test on fold deltas.

    Standard paired-t assumes independent observations; k-fold CV folds are
    NOT independent (each pair of folds' training sets overlap in all but
    1/k of their rows), which makes the naive paired-t anti-conservative.
    The correction inflates the variance estimate by ``n_test / n_train``
    (the fraction of data used for testing versus training in each fold),
    reflecting how much fold-to-fold correlation to expect.

    ``t = mean_delta / sqrt( s^2 * (1/k + n_test/n_train) )``, with k-1
    degrees of freedom, where s^2 is the sample variance of the deltas
    (ddof=1). A two-sided p-value is returned; callers wanting "does the
    better side actually win" combine this with a direction check on
    ``mean_delta``'s sign (one-sided in effect).

    Returns t_statistic=None, p_value=None when the deltas have zero
    variance (a degenerate case — no meaningful test) or fewer than 2 folds.
    """
    from scipy import stats

    k = len(deltas)
    if k < 2:
        return PairedTestResult(
            k=k, mean_delta=(deltas[0] if deltas else 0.0), std_delta=0.0,
            t_statistic=None, p_value=None, degrees_of_freedom=0,
        )
    mean_delta = sum(deltas) / k
    variance = sum((d - mean_delta) ** 2 for d in deltas) / (k - 1)  # ddof=1
    std_delta = math.sqrt(variance)
    df = k - 1

    if variance <= 1e-15 or n_train <= 0:
        return PairedTestResult(
            k=k, mean_delta=mean_delta, std_delta=std_delta,
            t_statistic=None, p_value=None, degrees_of_freedom=df,
        )

    correction = (1.0 / k) + (n_test / n_train)
    denom = math.sqrt(variance * correction)
    if denom <= 1e-15:
        return PairedTestResult(
            k=k, mean_delta=mean_delta, std_delta=std_delta,
            t_statistic=None, p_value=None, degrees_of_freedom=df,
        )
    t_statistic = mean_delta / denom
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=df))
    return PairedTestResult(
        k=k,
        mean_delta=mean_delta,
        std_delta=std_delta,
        t_statistic=t_statistic,
        p_value=p_value,
        degrees_of_freedom=df,
    )


def kfold_train_test_sizes(n_rows: int, n_folds: int) -> tuple[int, int]:
    """(n_train, n_test) for one fold of ordinary k-fold CV over n_rows."""
    n_test = round(n_rows / n_folds)
    n_train = n_rows - n_test
    return max(n_train, 1), max(n_test, 1)
