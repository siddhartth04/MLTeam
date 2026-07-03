"""Deterministic leakage detectors.

Three independent checks, all evidence-based:
1. near-perfect features — a single feature with near-perfect MI/correlation
   to the target almost always means the target leaked into the features;
2. holdout untouched — asserted from the store guard's access log;
3. suspicious perfection — bounded CV metrics at their ceiling with tiny
   fold variance are a red flag, not a triumph.
"""

from __future__ import annotations

from mlteam.contracts import DataProfile, LeakageCheck, MetricSummary, WarningCode
from mlteam.evaluation.metrics import PERFECTION_THRESHOLDS
from mlteam.store.guard import GuardedArtifactStore

_LEAK_MI_THRESHOLD = 0.9
_LEAK_CORR_THRESHOLD = 0.99


def check_near_perfect_features(profile: DataProfile) -> LeakageCheck:
    """Fail if any feature predicts the target almost perfectly."""
    suspects = [
        w.column
        for w in profile.warnings
        if w.code is WarningCode.SUSPECTED_LEAK and w.column is not None
    ]
    for rel in profile.target_relationships:
        mi_leak = rel.mutual_information is not None and rel.mutual_information > _LEAK_MI_THRESHOLD
        corr_leak = rel.correlation is not None and abs(rel.correlation) > _LEAK_CORR_THRESHOLD
        if (mi_leak or corr_leak) and rel.column not in suspects:
            suspects.append(rel.column)
    if suspects:
        return LeakageCheck(
            name="near_perfect_features",
            passed=False,
            details=f"features with near-perfect target signal: {sorted(set(suspects))}",
        )
    return LeakageCheck(
        name="near_perfect_features",
        passed=True,
        details="no feature exceeds MI/correlation leak thresholds",
    )


def check_holdout_untouched(guard: GuardedArtifactStore) -> LeakageCheck:
    """Assert from the guard's evidence log that the holdout was never read."""
    untouched = guard.holdout_untouched()
    denied = sum(1 for r in guard.access_log() if not r.granted)
    return LeakageCheck(
        name="holdout_untouched",
        passed=untouched,
        details=(f"guard evidence: untouched={untouched}, blocked/probed attempts={denied}"),
    )


def check_suspicious_perfection(primary: MetricSummary) -> LeakageCheck:
    """Fail if the primary metric sits at its ceiling — too good to trust."""
    threshold = PERFECTION_THRESHOLDS.get(primary.name)
    if threshold is not None and primary.mean >= threshold:
        return LeakageCheck(
            name="suspicious_perfection",
            passed=False,
            details=(
                f"{primary.name} mean {primary.mean:.4f} >= {threshold} with "
                f"fold std {primary.std:.5f} — investigate leakage before trusting this"
            ),
        )
    if primary.name in ("rmse", "mae", "log_loss") and primary.mean <= 1e-9:
        return LeakageCheck(
            name="suspicious_perfection",
            passed=False,
            details=f"{primary.name} is exactly zero across folds — almost certainly leakage",
        )
    return LeakageCheck(
        name="suspicious_perfection",
        passed=True,
        details=f"{primary.name} mean {primary.mean:.4f} is within plausible range",
    )
