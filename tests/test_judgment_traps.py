"""JUDGMENT_ROADMAP.md Task 1: the judgment trap suite as executable specs.

Each test drives the REAL deterministic pipeline (run_tabular_pipeline)
against one benchmarks/judgment_traps.py generator and asserts the single
senior-engineer call that trap is designed to demand. Six of eight traps
assert the call is made TODAY; the remaining blocked trap is marked
xfail(strict=True) and keyed to the roadmap task that closes it -- when
that task lands, its assertion starts passing and strict=True turns the
leftover xfail marker into a hard error, forcing the marker's removal in
the same commit.

temporal_drift_disguised has no task assigned (see JUDGMENT_ROADMAP.md's
deferred section) but is still asserted here in its intended, currently-
failing form: pinning today's gap as a documented, executable fact rather
than an unchecked assumption.
"""

from __future__ import annotations

import pytest
from benchmarks.judgment_traps import (
    constant_target,
    duplicate_rows,
    high_cardinality_categorical,
    id_memorization,
    leaky_feature,
    placeholder_labels,
    skewed_price_target,
    temporal_drift_disguised,
)

from mlteam.agents.data_io import read_dataset
from mlteam.contracts import TaskType, TriageIssueCode, WarningSeverity
from mlteam.orchestration import run_tabular_pipeline
from mlteam.store import GuardedArtifactStore, LocalArtifactStore

FAST_LGBM = {"n_estimators": 30}


def _fresh_store(root):
    return GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")


def _run(tmp_path, df, *, target_column, task_type, primary_metric, run_id):
    csv_path = tmp_path / "data.csv"
    df.write_csv(csv_path)
    store = _fresh_store(tmp_path / "store")
    result = run_tabular_pipeline(
        store,
        csv_path,
        target_column=target_column,
        task_type=task_type,
        primary_metric=primary_metric,
        run_id=run_id,
        seed=42,
        n_folds=3,
        include_tabpfn=False,
        lgbm_config=FAST_LGBM,
    )
    return store, result


# -- must-pass traps ----------------------------------------------------


def test_constant_target_fails_loudly_naming_the_target(tmp_path):
    """A zero-variance target must raise a clear ValueError naming the
    target column, not crash deep inside sklearn/cleaning (the exact bug
    this trap discovered and JUDGMENT_ROADMAP.md Task 1 fixed)."""
    with pytest.raises(ValueError, match="constant_price"):
        _run(
            tmp_path,
            constant_target(),
            target_column="constant_price",
            task_type=TaskType.REGRESSION,
            primary_metric="rmse",
            run_id="trap-constant-target",
        )


def test_leaky_feature_is_flagged_by_leakage_checks(tmp_path):
    """A near-perfect-MI leaked feature must be caught by the deterministic
    leakage detectors, not silently drive a too-good-to-trust score."""
    _, result = _run(
        tmp_path,
        leaky_feature(),
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id="trap-leaky-feature",
    )
    checks = {c.name: c for c in result.evaluation_report.leakage_checks}
    assert not checks["near_perfect_features"].passed
    assert "leaky_echo" in checks["near_perfect_features"].details
    assert not checks["suspicious_perfection"].passed


def test_id_memorization_drops_the_identifier_column(tmp_path):
    """A unique-per-row identifier must never reach training."""
    store, result = _run(
        tmp_path,
        id_memorization(),
        target_column="churned",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id="trap-id-memorization",
    )
    cleaned = read_dataset(store, store.get(result.cleaned_dataset_id))
    assert "customer_id" not in cleaned.columns


def test_skewed_price_target_gets_a_log1p_transform(tmp_path):
    """JUDGMENT_ROADMAP.md Task 2 flips this trap: the profiler recommends
    log1p for a target spanning multiple orders of magnitude, and the
    deterministic pipeline applies it to every regression baseline. The
    winner must carry target_transform=="log1p", beat an identically
    configured raw-target run (run both here — the comparison is the
    point), and report metrics in ORIGINAL units, not log-scale."""
    from mlteam.agents.trainer import compute_env_hash, train_experiment
    from mlteam.contracts import BudgetEstimate, ExperimentSpec

    store, result = _run(
        tmp_path,
        skewed_price_target(),
        target_column="price",
        task_type=TaskType.REGRESSION,
        primary_metric="mae",
        run_id="trap-skewed-price",
    )
    winner = store.get(result.evaluation_report.model_ref.artifact_id)
    assert winner.target_transform == "log1p"

    raw_spec = ExperimentSpec(
        experiment_id="raw-comparison",
        template_id=winner.template_id,
        config=dict(winner.config),
        budget=BudgetEstimate(wall_time_seconds=30),
        is_baseline=False,
        target_transform=None,
    )
    raw_result = train_experiment(
        store,
        result.cleaned_dataset_id,
        raw_spec,
        task_type=TaskType.REGRESSION,
        run_id="trap-skewed-price",
        seed=42,
        n_folds=3,
        env_hash=compute_env_hash(),
    )
    winner_mae = next(m for m in result.evaluation_report.cv_metrics if m.name == "mae").mean
    assert winner_mae < raw_result.metrics["mae"].mean

    winner_rmse = next(m for m in result.evaluation_report.cv_metrics if m.name == "rmse").mean
    assert winner_rmse > 1000  # original units -- log-scale rmse would be ~0-10


def test_duplicate_rows_are_deduped_before_split(tmp_path):
    """Exact-duplicate rows must be removed by triage, before any split can
    let an identical row land in both train and holdout."""
    store, result = _run(
        tmp_path,
        duplicate_rows(),
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id="trap-duplicate-rows",
    )
    triage = store.get(result.triage_report_id)
    codes = [i.code for i in triage.issues]
    assert TriageIssueCode.DUPLICATE_ROWS in codes
    assert triage.rows_before == 375
    assert triage.rows_after == 300


def test_placeholder_labels_are_coerced_and_named(tmp_path):
    """A mostly-numeric target with a marketplace placeholder string must
    be coerced, and the junk value must be named in the record -- not just
    a silent row drop."""
    store, result = _run(
        tmp_path,
        placeholder_labels(),
        target_column="price",
        task_type=TaskType.REGRESSION,
        primary_metric="rmse",
        run_id="trap-placeholder-labels",
    )
    triage = store.get(result.triage_report_id)
    codes = [i.code for i in triage.issues]
    assert TriageIssueCode.TARGET_NOT_NUMERIC in codes
    assert TriageIssueCode.TARGET_MISSING_ROWS in codes
    justifications = " ".join(r.justification for r in triage.remedies)
    assert "Call for Price" in justifications


def test_high_cardinality_categorical_beats_the_dummy_baseline(tmp_path):
    """JUDGMENT_ROADMAP.md Task 5 flips this trap: before cross-fitted
    target encoding, the best real model (whatever encoding it fell back to
    for a 150-level categorical) scored roc_auc 0.34-0.41 -- WORSE than the
    0.5 dummy/majority-class baseline from the SAME run, despite the column
    carrying real signal by construction. Comparing within one run needs no
    separate calibration run."""
    _, result = _run(
        tmp_path,
        high_cardinality_categorical(),
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id="trap-high-cardinality",
    )
    dummy_auc = result.train_results["exp-dummy"].metrics["roc_auc"].mean
    best_real_auc = max(
        tr.metrics["roc_auc"].mean
        for key, tr in result.train_results.items()
        if key != "exp-dummy"
    )
    assert best_real_auc > dummy_auc


# -- blocked traps (xfail, keyed to the roadmap task that closes them) --


@pytest.mark.xfail(
    strict=True,
    reason=(
        "no task assigned (see JUDGMENT_ROADMAP.md's deferred section): "
        "adversarial validation / temporal-risk detection needs real design. "
        "Today only an INFO-severity DATETIME_DETECTED profile warning "
        "fires; nothing evaluates whether random K-fold CV is actually "
        "safe under an unmarked regime shift."
    ),
)
def test_temporal_drift_disguised_warns_that_random_cv_is_unsafe(tmp_path):
    store, result = _run(
        tmp_path,
        temporal_drift_disguised(),
        target_column="y",
        task_type=TaskType.REGRESSION,
        primary_metric="rmse",
        run_id="trap-temporal-drift",
    )
    profile = store.get(result.raw_profile_id)
    assert any(
        w.severity in (WarningSeverity.WARNING, WarningSeverity.CRITICAL)
        and "temporal" in w.message.lower()
        for w in profile.warnings
    )
