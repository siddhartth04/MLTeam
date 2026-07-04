"""Metric definitions: task-appropriate scorers with explicit directions."""

from __future__ import annotations

from mlteam.contracts import MetricSummary, TaskType

# Our metric name -> sklearn scorer name. neg_* scorers are sign-flipped back
# to their natural positive form before reporting.
_SCORING: dict[TaskType, dict[str, str]] = {
    TaskType.BINARY_CLASSIFICATION: {
        "roc_auc": "roc_auc",
        # PR-AUC: the honest headline under class imbalance (fraud, churn).
        # NOTE: sklearn's average_precision scores classes_[1] as positive,
        # so 0/1 or False/True labels score the rare class as intended.
        "pr_auc": "average_precision",
        "accuracy": "accuracy",
        # f1_macro, not f1: the plain f1 scorer assumes pos_label=1 and
        # crashes on string labels ("good"/"bad"); macro is label-agnostic.
        "f1_macro": "f1_macro",
        "log_loss": "neg_log_loss",
    },
    TaskType.ANOMALY_DETECTION: {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1_macro": "f1_macro",
    },
    TaskType.MULTICLASS_CLASSIFICATION: {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "log_loss": "neg_log_loss",
    },
    TaskType.REGRESSION: {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    },
    # Forecasting is scored like regression; the difference is in splitting
    # (chronological) and features (lags), not in the error measures.
    TaskType.FORECASTING: {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    },
}

HIGHER_IS_BETTER: frozenset[str] = frozenset({"roc_auc", "pr_auc", "accuracy", "f1_macro", "r2"})

# Bounded metrics where a near-max mean is a leakage red flag, with thresholds.
PERFECTION_THRESHOLDS: dict[str, float] = {
    "roc_auc": 0.999,
    "pr_auc": 0.999,
    "accuracy": 0.999,
    "f1_macro": 0.999,
    "r2": 0.999,
}


def scoring_for(task_type: TaskType) -> dict[str, str]:
    return dict(_SCORING[task_type])


def is_higher_better(metric: str) -> bool:
    return metric in HIGHER_IS_BETTER


def to_natural_sign(metric: str, sklearn_scorer: str, values: list[float]) -> list[float]:
    """Flip neg_* scorer outputs back to the metric's natural positive form."""
    if sklearn_scorer.startswith("neg_"):
        return [-v for v in values]
    return values


def summarize_folds(name: str, fold_values: list[float]) -> MetricSummary:
    n = len(fold_values)
    mean = sum(fold_values) / n
    variance = sum((v - mean) ** 2 for v in fold_values) / n
    return MetricSummary(
        name=name,
        fold_values=tuple(round(v, 10) for v in fold_values),
        mean=round(mean, 10),
        std=round(variance**0.5, 10),
    )


def better_than(metric: str, candidate: float, reference: float) -> bool:
    if is_higher_better(metric):
        return candidate > reference
    return candidate < reference
