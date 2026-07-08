"""EvaluationReport builder: all numbers deterministic, evidence-backed.

``holdout_untouched`` is computed from the guard's access log here — the ONLY
sanctioned way to populate that field — never passed in by callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlteam.contracts import (
    BaselineDelta,
    CalibrationSummary,
    CostAsymmetry,
    DataProfile,
    DecisionThreshold,
    EvaluationReport,
    TaskType,
    is_classification,
    new_artifact,
)
from mlteam.evaluation.calibration import compute_calibration
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
from mlteam.evaluation.thresholds import cost_optimal_threshold
from mlteam.store.guard import GuardedArtifactStore

if TYPE_CHECKING:
    # mlteam.orchestration imports THIS module (agentic.py -> evaluation.report),
    # so importing mlteam.orchestration.progress at runtime here would be
    # circular -- ProgressReporter is only ever used as a type annotation,
    # and `from __future__ import annotations` means annotations are never
    # evaluated at runtime, so a TYPE_CHECKING-only import is sufficient.
    from mlteam.orchestration.progress import ProgressReporter

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


def _calibration_and_threshold(
    store: GuardedArtifactStore,
    *,
    dataset_id: str,
    model_result: TrainResult,
    task_type: TaskType,
    cost_asymmetry: CostAsymmetry | None,
    n_folds: int,
    seed: int,
    reporter: ProgressReporter | None = None,
) -> tuple[CalibrationSummary, DecisionThreshold | None]:
    """Best-effort: an OOF refit failing (e.g. unsupported task type, or the
    template lacks predict_proba) must never break report building — the
    rest of the EvaluationReport is far more load-bearing than these two
    fields, so log and return empty/None rather than raise.

    Skips the OOF refit entirely for non-classification tasks. Before this
    guard existed, a live regression run paid the FULL cost of refitting
    the winning model per fold here -- a 5-base-learner stacking ensemble,
    5 folds, 187k rows, ~25 silent minutes -- only to discover afterward
    that ``oof.y_proba is None`` (regression predictions have no positive-
    class probability) and discard the entire result. Calibration and
    decision thresholds are meaningful only for probabilistic classifiers;
    checking that analytically before refitting, rather than after, turns
    a wasted expensive computation into a no-op for every regression and
    forecasting run.
    """
    if not is_classification(task_type):
        return CalibrationSummary(), None

    # Deferred import: evaluation.oof -> agents.trainer/data_io ->
    # agents.__init__ -> agents.deployer -> orchestration.gates ->
    # orchestration.__init__ -> orchestration.agentic -> evaluation.report,
    # a real circular import at module load time if this were top-level.
    from mlteam.evaluation.oof import compute_oof

    try:
        oof = compute_oof(
            store,
            dataset_id,
            template_id=model_result.model.template_id,
            config=model_result.model.config,
            task_type=task_type,
            seed=seed,
            n_folds=n_folds,
            target_transform=model_result.model.target_transform,
            reporter=reporter,
        )
    except Exception:  # noqa: BLE001 - best-effort, never blocks the report
        return CalibrationSummary(), None
    if oof.y_proba is None:
        return CalibrationSummary(), None

    calib = compute_calibration(oof.y_true, oof.y_proba)
    calibration = CalibrationSummary(
        expected_calibration_error=calib.expected_calibration_error,
        brier_score=calib.brier_score,
        method=f"out_of_fold_binned_ece_{calib.n_bins}bins",
    )

    threshold: DecisionThreshold | None = None
    if cost_asymmetry is not None:
        result = cost_optimal_threshold(
            oof.y_true,
            oof.y_proba,
            false_positive_cost=cost_asymmetry.false_positive_cost,
            false_negative_cost=cost_asymmetry.false_negative_cost,
        )
        threshold = DecisionThreshold(
            threshold=result.threshold,
            expected_cost=result.expected_cost,
            cost_at_default_threshold=result.cost_at_default,
            false_positive_cost=result.false_positive_cost,
            false_negative_cost=result.false_negative_cost,
            false_positives=result.false_positives,
            false_negatives=result.false_negatives,
            true_positives=result.true_positives,
            true_negatives=result.true_negatives,
        )
    return calibration, threshold


def build_evaluation_report(
    store: GuardedArtifactStore,
    *,
    run_id: str,
    model_result: TrainResult,
    dummy_result: TrainResult,
    profile: DataProfile,
    primary_metric: str,
    n_rows: int | None = None,
    dataset_id: str | None = None,
    task_type: TaskType | None = None,
    cost_asymmetry: CostAsymmetry | None = None,
    n_folds: int = 5,
    seed: int = 42,
    reporter: ProgressReporter | None = None,
) -> EvaluationReport:
    """Assemble and store the EvaluationReport for the selected model.

    ``n_rows`` (the training dataset's row count) enables the real paired
    Nadeau-Bengio corrected significance test over per-fold deltas. When
    omitted (or when the model/baseline fold counts don't match, meaning the
    two were not scored on the same split), ``significant`` falls back to
    the documented pooled-std heuristic instead — see BaselineDelta's
    ``significance_method`` field for which path was taken.

    ``dataset_id``/``task_type`` (when both given AND the task is
    classification) drive an out-of-fold refit that measures REAL
    calibration (Brier/ECE) instead of the permanently-empty
    CalibrationSummary this used to always ship, and — when
    ``cost_asymmetry`` is also given — the cost-optimal decision threshold
    for the business's stated false-positive/false-negative cost ratio. Both
    are omitted (not a vacuous pass) when the prerequisites aren't met.
    Non-classification tasks skip the refit entirely rather than running it
    and discarding the result -- see ``_calibration_and_threshold``.

    ``reporter``, if given, receives one log line per CV fold as the OOF
    refit progresses (classification tasks only, since that's the only case
    where the refit actually runs) -- this stage previously refit the
    winning model with zero visible progress for however long the refit
    took.
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

    calibration = CalibrationSummary()
    decision_threshold: DecisionThreshold | None = None
    if dataset_id is not None and task_type is not None:
        calibration, decision_threshold = _calibration_and_threshold(
            store,
            dataset_id=dataset_id,
            model_result=model_result,
            task_type=task_type,
            cost_asymmetry=cost_asymmetry,
            n_folds=n_folds,
            seed=seed,
            reporter=reporter,
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
        calibration=calibration,
        decision_threshold=decision_threshold,
        leakage_checks=leakage_checks,
        label_issues=None,
        holdout_untouched=store.holdout_untouched(),
    )
    store.put(report)
    return report
