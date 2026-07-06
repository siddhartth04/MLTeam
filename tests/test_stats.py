"""Audit Task 2 (H1): paired Nadeau-Bengio corrected significance testing.

Hand-verified cases (values reproduced in each docstring so a reader can
check them with a calculator, not just trust the assertion).
"""

from __future__ import annotations

import math

import pytest

from mlteam.evaluation.stats import (
    fold_pairing_is_valid,
    kfold_train_test_sizes,
    nadeau_bengio_test,
    paired_fold_deltas,
)


def test_paired_fold_deltas_pairs_by_index():
    a = (0.62, 0.60, 0.61)
    b = (0.50, 0.51, 0.50)
    deltas = paired_fold_deltas(a, b)
    assert deltas == pytest.approx((0.12, 0.09, 0.11))


def test_paired_fold_deltas_rejects_mismatched_fold_counts():
    with pytest.raises(ValueError, match="cannot pair"):
        paired_fold_deltas((0.1, 0.2), (0.1, 0.2, 0.3))


def test_fold_pairing_is_valid_requires_matching_counts_and_at_least_two():
    assert fold_pairing_is_valid((0.1, 0.2), (0.3, 0.4)) is True
    assert fold_pairing_is_valid((0.1,), (0.3,)) is False  # k=1, no variance
    assert fold_pairing_is_valid((0.1, 0.2), (0.3, 0.4, 0.5)) is False


def test_kfold_train_test_sizes_ordinary_kfold():
    # 300 rows, 3 folds -> ~100 test, ~200 train per fold.
    n_train, n_test = kfold_train_test_sizes(300, 3)
    assert n_train == 200
    assert n_test == 100


def test_nadeau_bengio_hand_computed_case():
    """deltas = (0.02, 0.03, 0.01), k=3.
    mean_delta = 0.02 exactly.
    sample variance (ddof=1) = ((0.02-0.02)^2+(0.03-0.02)^2+(0.01-0.02)^2)/2
                              = (0 + 0.0001 + 0.0001)/2 = 0.0001 -> std=0.01.
    With n_train=200, n_test=100: correction = 1/3 + 100/200 = 0.8333...
    denom = sqrt(0.0001 * 0.8333...) = 0.009128709...
    t = 0.02 / 0.009128709 = 2.190890...
    df = 2; two-sided p from scipy.stats.t.sf(2.19089, df=2)*2 = 0.159832...
    (verified independently against scipy in the implementation notes)."""
    deltas = (0.02, 0.03, 0.01)
    result = nadeau_bengio_test(deltas, n_train=200, n_test=100)
    assert result.k == 3
    assert result.mean_delta == pytest.approx(0.02, abs=1e-9)
    assert result.std_delta == pytest.approx(0.01, abs=1e-9)
    assert result.degrees_of_freedom == 2
    assert result.t_statistic == pytest.approx(2.190890, abs=1e-5)
    assert result.p_value == pytest.approx(0.159832, abs=1e-5)


def test_nadeau_bengio_catches_a_consistent_effect_marginal_variance_hides():
    """The whole point of pairing: model and baseline both swing a lot
    fold-to-fold (hard vs. easy folds), so their MARGINAL (unpaired) std is
    large and a pooled-std heuristic calls the gap 'not significant' --
    but the model is consistently ~0.05 better on every single fold, which
    the PAIRED per-fold delta reveals with much smaller variance.

    model_folds = (0.56, 0.71, 0.84), base_folds = (0.50, 0.65, 0.80).
    Marginal means/stds (ddof=1): model mean=0.703333, std=0.140238;
    base mean=0.65, std=0.150000. delta=0.053333.
    Old heuristic (sqrt(sum of variances)): pooled=0.205640,
        threshold=2*0.205640=0.411281 -> 0.053333 > 0.411281 is False (NOT significant).
    New heuristic (sqrt(mean of variances)): pooled=0.145436,
        threshold=0.290871 -> still NOT significant.
    Paired deltas = (0.06, 0.06, 0.04), mean=0.053333, std(ddof=1)=0.011547.
    With n_train=200, n_test=100: t=5.059644, df=2, p=0.036913 < 0.05
        -> the paired test correctly detects the real, consistent effect.
    """
    model_folds = (0.56, 0.71, 0.84)
    base_folds = (0.50, 0.65, 0.80)
    k = 3

    def _mean(xs: tuple[float, ...]) -> float:
        return sum(xs) / len(xs)

    def _std(xs: tuple[float, ...]) -> float:
        m = _mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    mean_m, std_m = _mean(model_folds), _std(model_folds)
    mean_b, std_b = _mean(base_folds), _std(base_folds)
    delta = mean_m - mean_b

    # Old (pre-audit) heuristic: sqrt(sum of variances).
    old_pooled = math.sqrt(std_m**2 + std_b**2)
    assert not (delta > 2.0 * old_pooled)

    # New (post-Task-2) heuristic: sqrt(mean of variances) -- still misses it.
    new_pooled = math.sqrt((std_m**2 + std_b**2) / 2)
    assert not (delta > 2.0 * new_pooled)

    deltas = paired_fold_deltas(model_folds, base_folds)
    assert deltas == pytest.approx((0.06, 0.06, 0.04), abs=1e-9)
    n_train, n_test = kfold_train_test_sizes(k * 100, k)
    result = nadeau_bengio_test(deltas, n_train=n_train, n_test=n_test)
    assert result.p_value is not None
    assert result.p_value < 0.05
    assert result.mean_delta > 0


def test_nadeau_bengio_returns_none_for_zero_variance_deltas():
    """A perfectly consistent delta (zero fold-to-fold variance) has no
    meaningful t-statistic -- there is nothing to estimate uncertainty from.
    Callers must fall back to a documented heuristic in this case, not
    fabricate an artificial p=0."""
    result = nadeau_bengio_test((0.05, 0.05, 0.05), n_train=200, n_test=100)
    assert result.t_statistic is None
    assert result.p_value is None
    assert result.mean_delta == pytest.approx(0.05)


def test_nadeau_bengio_returns_none_below_two_folds():
    result = nadeau_bengio_test((0.05,), n_train=200, n_test=100)
    assert result.k == 1
    assert result.t_statistic is None
    assert result.p_value is None
