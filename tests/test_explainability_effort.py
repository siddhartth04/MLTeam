"""JUDGMENT_ROADMAP.md Task 6: value-of-computation v0 -- effort scales
with verdict.

Before this, a HALT/unshippable run paid the same ~hour of SHAP/PDP
explainability compute as an approved model bound for deployment
("polishing a turd", observed live). ``effort="cheap"`` skips SHAP/PDP
entirely and runs permutation importance with fewer repeats; the decision
of WHEN to go cheap (``_explainability_effort``) is tested separately from
the mechanical effect of the effort level on ``explain_model`` itself.

The log-line assertion ("the user must see WHY the report is thinner") is
covered in tests/test_halt_gating.py, which already has the harness to run
the real run_manager._execute with a stubbed HALT result.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlteam.agents.explainer import explain_model
from mlteam.contracts import CritiqueReport, Finding, FindingSeverity, Verdict, new_artifact

RUN = "test-explainability-effort"


def _critique(verdict: Verdict, findings: tuple[Finding, ...] = ()) -> CritiqueReport:
    return new_artifact(
        CritiqueReport,
        producer_agent="test",
        run_id=RUN,
        evaluation_ref={"artifact_id": "e", "artifact_type": "EvaluationReport"},
        verdict=verdict,
        findings=findings,
        iteration_actions=("x",) if verdict in (Verdict.HALT, Verdict.ITERATE) else (),
    )


def _fitness_finding() -> Finding:
    return Finding(
        severity=FindingSeverity.HIGH,
        category="fitness_for_purpose",
        evidence="relative error is 40% of the median target value",
        recommendation="ask the business for an error tolerance",
    )


@pytest.mark.parametrize(
    ("status", "verdict", "findings", "expected_effort"),
    [
        ("halted", Verdict.HALT, (), "cheap"),
        ("approved", Verdict.APPROVE, (), "full"),
        ("budget_exhausted", Verdict.ITERATE, (_fitness_finding(),), "cheap"),
        # ITERATE + budget-exhausted but WITHOUT the fitness finding: full.
        ("budget_exhausted", Verdict.ITERATE, (), "full"),
        # Fitness finding present, but the run isn't budget-exhausted
        # (still iterating normally) -- not this task's trigger condition.
        ("approved", Verdict.APPROVE, (_fitness_finding(),), "full"),
    ],
)
def test_explainability_effort_decision(status, verdict, findings, expected_effort):
    from mlteam.api.run_manager import _explainability_effort

    result = SimpleNamespace(status=status, final_critique=_critique(verdict, findings))
    effort, reason = _explainability_effort(result)
    assert effort == expected_effort
    if expected_effort == "cheap":
        assert reason
    else:
        assert reason is None


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from mlteam.contracts import TaskType
    from mlteam.orchestration import run_tabular_pipeline
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore
    from synthetic import make_churn_frame

    tmp = tmp_path_factory.mktemp("explain-effort")
    csv_path = tmp / "churn.csv"
    make_churn_frame(n_rows=300).write_csv(csv_path)
    root = tmp / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    result = run_tabular_pipeline(
        store,
        csv_path,
        target_column="churned",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id=RUN,
        seed=42,
        n_folds=3,
        include_tabpfn=False,
        lgbm_config={"n_estimators": 30},
    )
    best = result.train_results[
        next(k for k in result.train_results if k != "exp-dummy" and "dummy" not in k)
    ]
    return store, result, best.model


def test_cheap_effort_skips_shap_and_pdp_but_keeps_importances(trained):
    from mlteam.contracts import TaskType

    store, result, model = trained
    report = explain_model(
        store,
        model,
        train_dataset_id=result.cleaned_dataset_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        n_folds=3,
        run_id=RUN,
        effort="cheap",
    )
    assert len(report.importances) >= 1
    assert report.shap_available is False
    kinds = {f.kind for f in report.figures}
    assert "shap_summary" not in kinds
    assert "partial_dependence" not in kinds
    assert "permutation_importance" in kinds


def test_full_effort_is_the_default_and_unchanged(trained):
    """Approve-path behavior must be untouched: default effort is "full",
    matching every existing phase5 assertion (not re-asserted here to avoid
    duplicating tests/test_phase5_pipeline.py -- this just pins the default
    itself, since a wrong default would silently downgrade every ordinary run."""
    import inspect

    default_effort = inspect.signature(explain_model).parameters["effort"].default
    assert default_effort == "full"
