"""Evaluation: metrics, deterministic leakage checks, report assembly."""

from mlteam.evaluation.leakage import (
    check_holdout_untouched,
    check_near_perfect_features,
    check_suspicious_perfection,
)
from mlteam.evaluation.metrics import (
    HIGHER_IS_BETTER,
    better_than,
    is_higher_better,
    scoring_for,
    summarize_folds,
)
from mlteam.evaluation.report import build_evaluation_report

__all__ = [
    "HIGHER_IS_BETTER",
    "better_than",
    "build_evaluation_report",
    "check_holdout_untouched",
    "check_near_perfect_features",
    "check_suspicious_perfection",
    "is_higher_better",
    "scoring_for",
    "summarize_folds",
]
