"""Leakage detector tests — including the deliberately leaky fixture."""

from __future__ import annotations

from mlteam.agents import ingest_file, profile_dataset
from mlteam.contracts import MetricSummary, WarningCode
from mlteam.evaluation import (
    check_holdout_untouched,
    check_near_perfect_features,
    check_suspicious_perfection,
)

RUN = "test-leak-run"


def _profile(store, csv_path):
    dataset = ingest_file(store, csv_path, target_column="churned", run_id=RUN)
    return profile_dataset(store, dataset.meta.artifact_id, run_id=RUN, seed=42)


def test_leaky_fixture_fires_profile_warning(guarded_store, leaky_churn_csv):
    profile = _profile(guarded_store, leaky_churn_csv)
    leaks = [w for w in profile.warnings if w.code is WarningCode.SUSPECTED_LEAK]
    assert leaks, "leaked_score should be flagged as a suspected leak"
    assert any(w.column == "leaked_score" for w in leaks)


def test_leaky_fixture_fails_near_perfect_check(guarded_store, leaky_churn_csv):
    profile = _profile(guarded_store, leaky_churn_csv)
    check = check_near_perfect_features(profile)
    assert check.passed is False
    assert "leaked_score" in check.details


def test_clean_fixture_passes_near_perfect_check(guarded_store, churn_csv):
    profile = _profile(guarded_store, churn_csv)
    assert check_near_perfect_features(profile).passed is True


def test_suspicious_perfection_flags_ceiling_auc():
    perfect = MetricSummary(name="roc_auc", fold_values=(1.0, 1.0, 0.999), mean=0.9997, std=0.0005)
    assert check_suspicious_perfection(perfect).passed is False


def test_suspicious_perfection_accepts_normal_auc():
    normal = MetricSummary(name="roc_auc", fold_values=(0.81, 0.79, 0.84), mean=0.8133, std=0.02)
    assert check_suspicious_perfection(normal).passed is True


def test_suspicious_perfection_flags_zero_rmse():
    zero = MetricSummary(name="rmse", fold_values=(0.0, 0.0), mean=0.0, std=0.0)
    assert check_suspicious_perfection(zero).passed is False


def test_holdout_untouched_check_uses_guard_evidence(guarded_store, make_dataset):
    from mlteam.contracts import DatasetRole

    blob = guarded_store.put_blob(b"holdout")
    guarded_store.put(make_dataset(role=DatasetRole.HOLDOUT, blob_hash=blob))
    check = check_holdout_untouched(guarded_store)
    assert check.passed is True
    assert "untouched=True" in check.details
