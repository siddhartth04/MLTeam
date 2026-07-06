"""Token-efficient artifact views for prompts.

Agents never see raw data — only compact, structured summaries of artifacts.
These views keep prompts small and deterministic (sorted, rounded).
"""

from __future__ import annotations

from typing import Any

from mlteam.contracts import DataProfile, EvaluationReport

_MAX_COLUMNS = 60
_MAX_RELATIONSHIPS = 30


def profile_summary(profile: DataProfile) -> dict[str, Any]:
    """Compact JSON view of a DataProfile for LLM consumption."""
    columns = [
        {
            "name": c.name,
            "dtype": c.dtype,
            "cardinality": c.cardinality,
            "missing_pct": round(c.missing_pct, 2),
            **({"mean": round(c.mean, 4)} if c.mean is not None else {}),
            **({"std": round(c.std, 4)} if c.std is not None else {}),
            **({"top_values": list(c.top_values[:3])} if c.top_values else {}),
        }
        for c in profile.columns[:_MAX_COLUMNS]
    ]
    relationships = sorted(
        profile.target_relationships,
        key=lambda r: -(r.mutual_information or 0.0),
    )[:_MAX_RELATIONSHIPS]
    return {
        "dataset_artifact": profile.dataset_ref.artifact_id[:12],
        "n_columns": len(profile.columns),
        "columns": columns,
        "target_relationships": [
            {
                "column": r.column,
                "mutual_information": None
                if r.mutual_information is None
                else round(r.mutual_information, 4),
                "correlation": None if r.correlation is None else round(r.correlation, 4),
            }
            for r in relationships
        ],
        "warnings": [
            {
                "code": w.code.value,
                "severity": w.severity.value,
                "column": w.column,
                "message": w.message,
            }
            for w in profile.warnings
        ],
    }


def evaluation_summary(report: EvaluationReport) -> dict[str, Any]:
    """Compact JSON view of an EvaluationReport — the ONLY thing the Critic sees."""
    return {
        "cv_metrics": [
            {
                "name": m.name,
                "mean": round(m.mean, 6),
                "std": round(m.std, 6),
                "fold_values": [round(v, 6) for v in m.fold_values],
            }
            for m in report.cv_metrics
        ],
        "baseline_deltas": [
            {
                "baseline": d.baseline_name,
                "metric": d.metric,
                "baseline_value": round(d.baseline_value, 6),
                "model_value": round(d.model_value, 6),
                "delta": round(d.delta, 6),
                "significant": d.significant,
                "p_value": d.p_value,
                "significance_method": d.significance_method,
            }
            for d in report.baseline_deltas
        ],
        "leakage_checks": [
            {"name": c.name, "passed": c.passed, "details": c.details}
            for c in report.leakage_checks
        ],
        "calibration": {
            "expected_calibration_error": report.calibration.expected_calibration_error,
            "brier_score": report.calibration.brier_score,
        },
        "label_issues": None
        if report.label_issues is None
        else {
            "n_flagged": report.label_issues.n_flagged,
            "flagged_fraction": report.label_issues.flagged_fraction,
        },
        "slice_metrics": [
            {"slice": s.slice_name, "n_rows": s.n_rows, "metrics": s.metrics}
            for s in report.slice_metrics
        ],
        "holdout_untouched": report.holdout_untouched,
    }
