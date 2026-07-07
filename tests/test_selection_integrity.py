"""Audit Task 8 (M1, M2): screening models compete for selection; the
deployment dossier's margin matches the model actually deployed.

Root cause (M2): _run_screening's trained results were discarded from the
candidate pool -- only surfaced as plain-dict evidence for the LLM planner's
payload. A screening probe (dummy/linear/lightgbm, trained BEFORE the plan
is drafted) could genuinely outscore every planned candidate while being
structurally unselectable, and would still appear in the outcome table the
critic reads (skewing compute_winner_margin negative for no good reason).

Root cause (M1): _select_best takes the max of CV means over an
ever-growing field (winner's curse grows with iterations/screening), and
nothing disclosed how many candidates the winner was chosen among.
Separately, the deployment dossier computed winner_margin from ranked[0]
over ALL completed outcomes while is_winner marked model.experiment_id --
when they differ, Gate 3's margin table could describe a DIFFERENT model
than the one being signed off (covered directly in
test_deployment_dossier.py; this file focuses on the pipeline-level
selection integrity).
"""

from __future__ import annotations

from pathlib import Path

from mlteam.contracts import Verdict
from mlteam.llm import FixtureTransport, LLMClient
from mlteam.orchestration import AutoApproveGate, run_agentic_pipeline
from mlteam.store import GuardedArtifactStore, LocalArtifactStore
from synthetic import make_churn_frame

FIXTURE = Path(__file__).parent / "fixtures" / "llm" / "agentic_churn_run.json"


def test_screening_winner_is_reflected_in_pipeline_result(tmp_path):
    """On the recorded agentic_churn_run fixture, screen-linear (roc_auc
    0.772) genuinely outscores every planned lightgbm candidate (0.769,
    0.748) -- PipelineResult must reflect linear_default as the winner, not
    silently exclude the screening probe from being selectable."""
    csv_path = tmp_path / "churn.csv"
    make_churn_frame().write_csv(csv_path)
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    client = LLMClient(FixtureTransport.from_file(FIXTURE), model="claude-sonnet-4-6")

    result = run_agentic_pipeline(
        store,
        csv_path,
        goal_text="Identify customers likely to churn so retention can call them first.",
        client=client,
        gate=AutoApproveGate(),
        seed=42,
        n_folds=3,
    )

    pipeline = result.pipeline
    assert pipeline.best_template_id == "linear_default"
    # The winning screening result must be present in train_results (not
    # excluded from the selectable pool) alongside the planned candidates.
    assert "screen-linear" in pipeline.train_results
    assert "exp-lgbm-default" in pipeline.train_results
    # The plan's own dummy stays authoritative for the dummy floor, distinct
    # from screening's dummy (both share template_id="dummy_baseline", so
    # this field is what disambiguates them for downstream code).
    assert pipeline.dummy_experiment_id == "exp-dummy"
    assert pipeline.train_results[pipeline.dummy_experiment_id].model.template_id == (
        "dummy_baseline"
    )


def test_evaluation_report_scores_the_actual_screening_winner(tmp_path):
    """The EvaluationReport (and therefore the critic's evidence) must be
    built from the SCREENING winner's TrainResult when it wins, not the
    best of only the planned candidates."""
    csv_path = tmp_path / "churn.csv"
    make_churn_frame().write_csv(csv_path)
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    client = LLMClient(FixtureTransport.from_file(FIXTURE), model="claude-sonnet-4-6")

    result = run_agentic_pipeline(
        store,
        csv_path,
        goal_text="Identify customers likely to churn so retention can call them first.",
        client=client,
        gate=AutoApproveGate(),
        seed=42,
        n_folds=3,
    )
    report = result.pipeline.evaluation_report
    # The report's model_ref must point at the screen-linear model, not one
    # of the planned lightgbm candidates.
    winning_model = result.pipeline.train_results["screen-linear"].model
    assert report.model_ref.artifact_id == winning_model.meta.artifact_id
    assert result.critique.verdict in (Verdict.APPROVE, Verdict.ITERATE, Verdict.HALT)
