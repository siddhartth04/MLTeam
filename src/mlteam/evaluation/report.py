"""EvaluationReport builder: all numbers deterministic, evidence-backed.

``holdout_untouched`` is computed from the guard's access log here — the ONLY
sanctioned way to populate that field — never passed in by callers.
"""

from __future__ import annotations

from mlteam.contracts import (
    BaselineDelta,
    DataProfile,
    EvaluationReport,
    new_artifact,
)
from mlteam.evaluation.leakage import (
    check_holdout_untouched,
    check_near_perfect_features,
    check_suspicious_perfection,
)
from mlteam.evaluation.metrics import better_than, is_higher_better
from mlteam.evaluation.results import TrainResult
from mlteam.store.guard import GuardedArtifactStore

_SIGNIFICANCE_STD_MULTIPLE = 2.0


def _baseline_deltas(
    model: TrainResult, baseline: TrainResult, baseline_name: str
) -> tuple[BaselineDelta, ...]:
    deltas = []
    for name, summary in model.metrics.items():
        reference = baseline.metrics.get(name)
        if reference is None:
            continue
        delta = summary.mean - reference.mean
        improvement = delta if is_higher_better(name) else -delta
        # Pooled fold variance: both the model's and the baseline's noise
        # must be small relative to the improvement to call it significant.
        pooled_std = (summary.std**2 + reference.std**2) ** 0.5
        deltas.append(
            BaselineDelta(
                baseline_name=baseline_name,
                metric=name,
                baseline_value=reference.mean,
                model_value=summary.mean,
                delta=round(delta, 10),
                significant=bool(
                    better_than(name, summary.mean, reference.mean)
                    and improvement > _SIGNIFICANCE_STD_MULTIPLE * pooled_std
                ),
            )
        )
    return tuple(deltas)


def build_evaluation_report(
    store: GuardedArtifactStore,
    *,
    run_id: str,
    model_result: TrainResult,
    dummy_result: TrainResult,
    profile: DataProfile,
    primary_metric: str,
) -> EvaluationReport:
    """Assemble and store the EvaluationReport for the selected model."""
    primary = model_result.metrics.get(primary_metric)
    if primary is None:
        available = sorted(model_result.metrics)
        raise ValueError(f"primary metric '{primary_metric}' not computed; have {available}")

    leakage_checks = (
        check_near_perfect_features(profile),
        check_suspicious_perfection(primary),
        check_holdout_untouched(store),
    )
    report = new_artifact(
        EvaluationReport,
        producer_agent="evaluator",
        run_id=run_id,
        parent_ids=(
            model_result.model.meta.artifact_id,
            dummy_result.model.meta.artifact_id,
            profile.meta.artifact_id,
        ),
        model_ref=model_result.model.ref(),
        cv_metrics=tuple(model_result.metrics.values()),
        baseline_deltas=_baseline_deltas(model_result, dummy_result, "dummy_baseline"),
        slice_metrics=(),
        leakage_checks=leakage_checks,
        label_issues=None,
        holdout_untouched=store.holdout_untouched(),
    )
    store.put(report)
    return report
