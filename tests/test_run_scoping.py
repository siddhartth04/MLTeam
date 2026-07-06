"""Audit Task 1 (C1): every run-scoped artifact query must use the
run_artifacts membership table, not first-writer ``artifacts.run_id``
attribution.

Root cause reproduced here directly (no full pipeline needed):
``new_artifact`` content-addresses on payload + config, deliberately
EXCLUDING run_id. Two runs that produce a byte-identical ExperimentOutcome
(same experiment_id/template_id/state/config/metrics/wall_time/error --
e.g. a screening probe re-seeded against the same tiny dataset) therefore
mint the SAME artifact id. ``lineage.record`` is ``INSERT OR IGNORE`` on the
``artifacts`` table, so only the FIRST run's run_id sticks there forever.
Before this fix, every run-scoped view filtered on that stale column and a
second run would see holes -- this is the exact live failure observed
during the audit: a healthy run halted with "missing outcomes for:
['screen-linear', 'exp-dummy']" because those two experiments had already
been produced, byte-identical, by an earlier seeding.
"""

from __future__ import annotations

from mlteam.api.views import (
    critique_view,
    experiments_view,
    explainability_view,
    problem_spec_view,
)
from mlteam.contracts import (
    ExperimentState,
    MetricSummary,
    new_artifact,
)
from mlteam.evaluation.field_checks import audit_terminal_states
from mlteam.evaluation.results import list_experiment_outcomes, store_experiment_outcome
from mlteam.store import GuardedArtifactStore, LocalArtifactStore


def _metric(name: str, mean: float, std: float) -> MetricSummary:
    return MetricSummary(name=name, fold_values=(mean - std, mean + std), mean=mean, std=std)


def test_identical_outcome_across_two_runs_is_visible_to_both(tmp_path):
    """The exact collision: run-1 and run-2 each produce a byte-identical
    'screen-dummy' outcome (same experiment_id/template/state/config/metrics/
    wall_time/error -> same content-addressed artifact id). Both runs must
    see it in their own list_experiment_outcomes -- not just run-1."""
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")

    outcome_1 = store_experiment_outcome(
        store,
        experiment_id="screen-dummy",
        template_id="dummy_baseline",
        state=ExperimentState.COMPLETED,
        run_id="run-1",
        config={},
        wall_time_seconds=0.0,
    )
    outcome_2 = store_experiment_outcome(
        store,
        experiment_id="screen-dummy",
        template_id="dummy_baseline",
        state=ExperimentState.COMPLETED,
        run_id="run-2",
        config={},
        wall_time_seconds=0.0,
    )
    # Confirms the dedup premise: identical content -> identical artifact id.
    assert outcome_1.meta.artifact_id == outcome_2.meta.artifact_id

    run1_ids = {o.experiment_id for o in list_experiment_outcomes(store, "run-1")}
    run2_ids = {o.experiment_id for o in list_experiment_outcomes(store, "run-2")}
    assert "screen-dummy" in run1_ids
    assert "screen-dummy" in run2_ids


def test_terminal_state_audit_does_not_false_halt_on_reused_outcome(tmp_path):
    """The critic's complete_experiment_field hard gate must not see a hole
    when a planned experiment's outcome was first produced by an earlier run
    but is byte-identical (and therefore reused) in this one."""
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")

    store_experiment_outcome(
        store,
        experiment_id="screen-dummy",
        template_id="dummy_baseline",
        state=ExperimentState.COMPLETED,
        run_id="run-1",
        config={},
        wall_time_seconds=0.0,
    )
    store_experiment_outcome(
        store,
        experiment_id="screen-dummy",
        template_id="dummy_baseline",
        state=ExperimentState.COMPLETED,
        run_id="run-2",
        config={},
        wall_time_seconds=0.0,
    )
    store_experiment_outcome(
        store,
        experiment_id="screen-linear",
        template_id="linear_default",
        state=ExperimentState.COMPLETED,
        run_id="run-2",
        config={"unique_to_run2": True},
        wall_time_seconds=1.2,
    )

    run2_outcomes = list_experiment_outcomes(store, "run-2")
    audit = audit_terminal_states(("screen-dummy", "screen-linear"), run2_outcomes)
    assert audit.complete, f"unexpected missing ids: {audit.missing_ids}"


def test_experiments_view_leaderboard_populated_for_second_run(tmp_path, guarded_store):
    """experiments_view (the dashboard leaderboard) must show the full field
    for a run that reused an earlier run's byte-identical outcome."""
    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")

    store_experiment_outcome(
        store,
        experiment_id="exp-dummy",
        template_id="dummy_baseline",
        state=ExperimentState.COMPLETED,
        run_id="run-1",
        config={},
        wall_time_seconds=0.0,
    )
    store_experiment_outcome(
        store,
        experiment_id="exp-dummy",
        template_id="dummy_baseline",
        state=ExperimentState.COMPLETED,
        run_id="run-2",
        config={},
        wall_time_seconds=0.0,
    )

    rows = experiments_view(store, "run-2")
    assert any(r["experiment_id"] == "exp-dummy" for r in rows)


def test_critique_problem_spec_explainability_views_scoped_by_membership(tmp_path):
    """critique_view/problem_spec_view/explainability_view must all use the
    membership table, not first-writer run_id, for byte-identical artifacts
    shared across two runs."""
    from mlteam.contracts import (
        CritiqueReport,
        ExplainabilityReport,
        FeatureImportance,
        ImportanceMethod,
        ProblemSpec,
        TaskType,
        Verdict,
    )

    root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")

    def _put_identical_pair(factory) -> None:
        a = factory(run_id="run-1")
        store.put(a)
        b = factory(run_id="run-2")
        assert a.meta.artifact_id == b.meta.artifact_id
        store.put(b)

    _put_identical_pair(
        lambda run_id: new_artifact(
            ProblemSpec,
            producer_agent="test",
            run_id=run_id,
            task_type=TaskType.BINARY_CLASSIFICATION,
            target_column="target",
            primary_metric="roc_auc",
            human_confirmed=True,
        )
    )
    _put_identical_pair(
        lambda run_id: new_artifact(
            CritiqueReport,
            producer_agent="test",
            run_id=run_id,
            evaluation_ref={"artifact_id": "e", "artifact_type": "EvaluationReport"},
            findings=(),
            verdict=Verdict.APPROVE,
            iteration_actions=(),
        )
    )
    _put_identical_pair(
        lambda run_id: new_artifact(
            ExplainabilityReport,
            producer_agent="test",
            run_id=run_id,
            model_ref={"artifact_id": "m", "artifact_type": "TrainedModel"},
            method=ImportanceMethod.PERMUTATION,
            importances=(FeatureImportance(feature="f1", importance=0.5, std=0.01),),
            shap_available=False,
            figures=(),
        )
    )

    assert problem_spec_view(store, "run-2") is not None
    assert critique_view(store, "run-2") is not None
    assert explainability_view(store, "run-2") is not None
