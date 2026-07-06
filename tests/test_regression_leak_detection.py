"""Audit Task 3 (C2, M5): regression leak detection.

Root cause: mutual_info_regression's raw output is differential-entropy-
based (nats) and UNBOUNDED for continuous targets -- unlike the
classification branch, which normalizes by target entropy into [0, 1]. A
legitimately strong (but non-leaky) predictor with r~=0.9 has MI~=0.83 nats,
comfortably above the un-normalized 0.9 threshold, producing a false
CRITICAL leak warning and a forced HALT via the near_perfect_features hard
gate. The fix converts regression MI to the Gaussian-equivalent correlation
sqrt(1 - exp(-2*MI)) (Joe, 1989), bounding it to the same [0,1) scale the
classification branch already uses, and adds Spearman correlation to catch
monotone-but-nonlinear leaks (e.g. exp(target)) that Pearson misses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mlteam.agents import ingest_file, profile_dataset
from mlteam.contracts import WarningCode
from mlteam.evaluation import check_near_perfect_features

RUN = "test-regression-leak"


def _write_csv(path: Path, df: pl.DataFrame) -> Path:
    df.write_csv(path)
    return path


def _profile_for(store, csv_path: Path, target_column: str, run_id: str):
    dataset = ingest_file(store, csv_path, target_column=target_column, run_id=run_id)
    return profile_dataset(store, dataset.meta.artifact_id, run_id=run_id, seed=42)


@pytest.fixture()
def strong_but_legit_regression_csv(tmp_path: Path) -> Path:
    """A feature with r ~= 0.9 to the target -- a genuinely strong, common,
    NON-leaky predictor (think: square footage vs. house price). The old
    unnormalized-MI check would flag this as a critical leak."""
    rng = np.random.default_rng(7)
    n = 400
    x1 = rng.normal(size=n)
    noise = rng.normal(size=n) * np.sqrt(1 - 0.9**2) / 0.9  # r(x1, y) ~= 0.9
    y = x1 + noise
    x2 = rng.normal(size=n)  # irrelevant filler feature
    df = pl.DataFrame({"strong_feature": x1, "filler": x2, "price": y})
    return _write_csv(tmp_path / "strong.csv", df)


@pytest.fixture()
def near_duplicate_regression_csv(tmp_path: Path) -> Path:
    """target + tiny noise: a genuine, obvious leak (r ~= 0.999)."""
    rng = np.random.default_rng(7)
    n = 400
    y = rng.normal(size=n) * 100
    leaked = y + rng.normal(size=n) * 0.01  # r ~= 0.999+
    filler = rng.normal(size=n)
    df = pl.DataFrame({"leaked_score": leaked, "filler": filler, "price": y})
    return _write_csv(tmp_path / "leaky.csv", df)


@pytest.fixture()
def monotone_nonlinear_leak_csv(tmp_path: Path) -> Path:
    """exp(target): perfectly monotone (spearman ~= 1.0) but only weakly
    LINEAR (pearson well below the leak threshold) -- Pearson alone misses
    this; Spearman catches it."""
    rng = np.random.default_rng(7)
    n = 400
    y = rng.uniform(1.0, 5.0, size=n)
    leaked = np.exp(y)
    filler = rng.normal(size=n)
    df = pl.DataFrame({"exp_of_target": leaked, "filler": filler, "target": y})
    return _write_csv(tmp_path / "monotone.csv", df)


def test_strong_legitimate_predictor_is_not_flagged_as_leak(
    guarded_store, strong_but_legit_regression_csv
):
    profile = _profile_for(guarded_store, strong_but_legit_regression_csv, "price", RUN)
    rel = next(r for r in profile.target_relationships if r.column == "strong_feature")
    # r~=0.9 -> Gaussian-equivalent MI ~= sqrt(1-exp(-2*0.83)) ~= 0.83, well
    # under the 0.9 threshold used for classification's entropy-normalized MI.
    assert rel.mutual_information is not None
    assert rel.mutual_information < 0.9, (
        f"legitimate r~=0.9 predictor's normalized MI ({rel.mutual_information}) "
        "should NOT cross the leak threshold"
    )
    assert not any(
        w.code is WarningCode.SUSPECTED_LEAK and w.column == "strong_feature"
        for w in profile.warnings
    )
    assert check_near_perfect_features(profile).passed is True


def test_near_duplicate_target_is_flagged_as_leak(guarded_store, near_duplicate_regression_csv):
    profile = _profile_for(guarded_store, near_duplicate_regression_csv, "price", RUN)
    rel = next(r for r in profile.target_relationships if r.column == "leaked_score")
    assert rel.correlation is not None
    assert abs(rel.correlation) > 0.99
    assert any(
        w.code is WarningCode.SUSPECTED_LEAK and w.column == "leaked_score"
        for w in profile.warnings
    )
    assert check_near_perfect_features(profile).passed is False


def test_monotone_nonlinear_leak_caught_by_spearman_not_pearson(
    guarded_store, monotone_nonlinear_leak_csv
):
    profile = _profile_for(guarded_store, monotone_nonlinear_leak_csv, "target", RUN)
    rel = next(r for r in profile.target_relationships if r.column == "exp_of_target")

    assert rel.spearman is not None
    assert abs(rel.spearman) > 0.99, "Spearman should catch the perfect monotone relationship"

    # Pearson alone (linear correlation of a highly nonlinear exp() transform)
    # should NOT be the thing catching this -- confirms Spearman is doing
    # genuinely additional work, not just agreeing with Pearson.
    assert rel.correlation is not None
    assert abs(rel.correlation) < 0.99, (
        "this test's premise requires Pearson to MISS the leak so Spearman "
        "demonstrably adds detection power"
    )

    assert any(
        w.code is WarningCode.SUSPECTED_LEAK and w.column == "exp_of_target"
        for w in profile.warnings
    )
    assert check_near_perfect_features(profile).passed is False
