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
from mlteam.evaluation.metrics import better_than, is_higher_better, signed_improvement
from mlteam.evaluation.results import TrainResult
from mlteam.evaluation.stats import (
    fold_pairing_is_valid,
    kfold_train_test_sizes,
    nadeau_bengio_test,
)
from mlteam.store.guard import GuardedArtifactStore

# Fallback heuristic significance threshold, used ONLY when a real paired
# test cannot be run (fold counts differ between model and baseline, or
# n_rows is not supplied). This is deliberately the same constant the
# heuristic has always used, so existing behavior is preserved exactly when
# the real test's preconditions are not met.
_HEURISTIC_STD_MULTIPLE = 2.0
_ALPHA = 0.05


def _heuristic_significance(
    name: str, summary_mean: float, summary_std: float, reference_mean: float, reference_std: float
) -> bool:
    """Fallback used only when the real paired test's preconditions aren't
    met. Pooled-std effect size, computed the SAME way field_checks.py's
    compute_winner_margin does (sqrt(mean of variances), not sqrt(sum of
    variances)) -- so a "how many pooled-sigma apart" number means the same
    thing everywhere in the platform, not one thing here and sqrt(2)x that
    in the critic's dossier."""
    delta = summary_mean - reference_mean
    improvement = delta if is_higher_better(name) else -delta
    pooled_std = ((summary_std**2 + reference_std**2) / 2) ** 0.5
    return bool(
        better_than(name, summary_mean, reference_mean)
        and improvement > _HEURISTIC_STD_MULTIPLE * pooled_std
    )


def _baseline_deltas(
    model: TrainResult,
    baseline: TrainResult,
    baseline_name: str,
    *,
    n_rows: int | None,
) -> tuple[BaselineDelta, ...]:
    deltas = []
    for name, summary in model.metrics.items():
        reference = baseline.metrics.get(name)
        if reference is None:
            continue
        delta = summary.mean - reference.mean

        p_value: float | None = None
        method = "heuristic_2sigma"
        significant: bool

        if n_rows is not None and fold_pairing_is_valid(
            summary.fold_values, reference.fold_values
        ):
            # Per-fold deltas in the metric's improve-is-positive direction,
            # so mean_delta > 0 always means "model beat baseline".
            model_folds = summary.fold_values
            ref_folds = reference.fold_values
            paired = tuple(
                signed_improvement(name, m, r) for m, r in zip(model_folds, ref_folds, strict=True)
            )
            n_train, n_test = kfold_train_test_sizes(n_rows, len(paired))
            test_result = nadeau_bengio_test(paired, n_train=n_train, n_test=n_test)
            if test_result.p_value is not None:
                p_value = round(test_result.p_value, 10)
                method = "nadeau_bengio_paired_t"
                significant = bool(test_result.mean_delta > 0 and test_result.p_value < _ALPHA)
            else:
                # Degenerate (zero-variance) deltas: no test statistic, but a
                # strictly positive mean delta with truly zero fold variance
                # on both sides is as unambiguous as significance testing
                # gets — fall through to the heuristic for a documented call.
                significant = _heuristic_significance(
                    name, summary.mean, summary.std, reference.mean, reference.std
                )
        else:
            significant = _heuristic_significance(
                name, summary.mean, summary.std, reference.mean, reference.std
            )

        deltas.append(
            BaselineDelta(
                baseline_name=baseline_name,
                metric=name,
                baseline_value=reference.mean,
                model_value=summary.mean,
                delta=round(delta, 10),
                significant=significant,
                p_value=p_value,
                significance_method=method,
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
    n_rows: int | None = None,
) -> EvaluationReport:
    """Assemble and store the EvaluationReport for the selected model.

    ``n_rows`` (the training dataset's row count) enables the real paired
    Nadeau-Bengio corrected significance test over per-fold deltas. When
    omitted (or when the model/baseline fold counts don't match, meaning the
    two were not scored on the same split), ``significant`` falls back to
    the documented pooled-std heuristic instead — see BaselineDelta's
    ``significance_method`` field for which path was taken.
    """
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
        baseline_deltas=_baseline_deltas(
            model_result, dummy_result, "dummy_baseline", n_rows=n_rows
        ),
        slice_metrics=(),
        leakage_checks=leakage_checks,
        label_issues=None,
        holdout_untouched=store.holdout_untouched(),
    )
    store.put(report)
    return report
