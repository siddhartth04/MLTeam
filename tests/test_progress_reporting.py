"""Phase 1 (dashboard UX): the pipeline must emit real per-stage progress as
it runs, not a batch flip at the end. A recording stub reporter drives the
same recorded-fixture pipeline used by test_agentic_pipeline.py and asserts
every deterministic stage goes active -> done, in canonical order, with at
least one log line each — and that replaying the fixture required no extra
LLM calls (call count/order is unchanged from the non-reporter run)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from mlteam.llm import FixtureTransport, LLMClient
from mlteam.orchestration import AutoApproveGate, run_agentic_pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "llm" / "agentic_churn_run.json"

_EXPECTED_STAGE_ORDER = (
    "requirement",
    "ingest",
    "profile",
    "split",
    "clean",
    "features",
    "plan",
    "train",
    "evaluate",
    "critic",
)


@dataclass
class RecordingReporter:
    stage_calls: list[tuple[str, str, str]] = field(default_factory=list)
    log_calls: list[tuple[str, str, str | None]] = field(default_factory=list)

    def stage(self, stage: str, status: str, detail: str = "") -> None:
        self.stage_calls.append((stage, status, detail))

    def log(self, message: str, *, level: str = "info", stage: str | None = None) -> None:
        self.log_calls.append((message, level, stage))


def _run(reporter: RecordingReporter | None):
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore
    from synthetic import make_churn_frame

    tmp = Path(tempfile.mkdtemp())
    csv_path = tmp / "churn.csv"
    make_churn_frame().write_csv(csv_path)
    root = tmp / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    client = LLMClient(FixtureTransport.from_file(FIXTURE), model="claude-sonnet-4-6")
    kwargs = dict(
        goal_text="Identify customers likely to churn so retention can call them first.",
        client=client,
        gate=AutoApproveGate(),
        seed=42,
        n_folds=3,
    )
    if reporter is not None:
        kwargs["reporter"] = reporter
    result = run_agentic_pipeline(store, csv_path, **kwargs)
    return result, client


def test_reporter_sees_every_stage_active_then_done_in_order():
    reporter = RecordingReporter()
    _run(reporter)

    seen_order = []
    active_stages = set()
    done_stages = set()
    for stage, status, _detail in reporter.stage_calls:
        if stage not in seen_order:
            seen_order.append(stage)
        if status == "active":
            active_stages.add(stage)
        elif status == "done":
            done_stages.add(stage)

    assert seen_order == list(_EXPECTED_STAGE_ORDER)
    assert active_stages == set(_EXPECTED_STAGE_ORDER)
    assert done_stages == set(_EXPECTED_STAGE_ORDER)

    # Each stage must go active before done (never done-before-active).
    first_active_index = {}
    first_done_index = {}
    for i, (stage, status, _detail) in enumerate(reporter.stage_calls):
        if status == "active" and stage not in first_active_index:
            first_active_index[stage] = i
        if status == "done" and stage not in first_done_index:
            first_done_index[stage] = i
    for stage in _EXPECTED_STAGE_ORDER:
        assert first_active_index[stage] < first_done_index[stage]


def test_reporter_logs_at_least_one_line_per_stage():
    reporter = RecordingReporter()
    _run(reporter)

    logged_stages = {stage for _msg, _level, stage in reporter.log_calls if stage is not None}
    # requirement's log call omits a stage kwarg (logged before any stage
    # concept applies cleanly) -- every OTHER stage logs at least one line.
    for stage in _EXPECTED_STAGE_ORDER[1:]:
        assert stage in logged_stages, f"no log line recorded for stage '{stage}'"


def test_reporter_logs_a_line_per_experiment_during_training():
    """A single house-prices run spent 59 silent minutes in the 'train'
    stage (four experiments, one a 5-base-learner stacking ensemble on
    187k rows) with only ONE log line at the very end -- the dashboard's
    activity feed showed no distinction between "training" and "hung".
    Every planned experiment (screening probes included, since they also
    run under _train_all) must now log at least a start line and a
    resolution line (completed/reused/failed), not just one summary at
    stage completion."""
    reporter = RecordingReporter()
    result, _client = _run(reporter)

    train_logs = [msg for msg, _level, stage in reporter.log_calls if stage == "train"]
    # Screening probes (screen-dummy, screen-linear, screen-lightgbm) plus
    # the plan's own experiments (exp-dummy, exp-lgbm-default, exp-lgbm-tuned)
    # all train under _train_all and must each get their own log lines --
    # PipelineResult.train_results is the merged screening+plan dict.
    all_experiment_ids = set(result.pipeline.train_results)
    assert all_experiment_ids  # sanity: the fixture actually trained something
    for experiment_id in all_experiment_ids:
        matching = [line for line in train_logs if experiment_id in line]
        assert matching, f"no 'train' stage log line mentions {experiment_id}"

    # More than one line per experiment (start + resolution), not a single
    # end-of-stage summary -- that's the whole point of this fix.
    assert len(train_logs) > len(all_experiment_ids)


def test_reporter_does_not_change_llm_call_count_or_order():
    """Fixture-replay safety: adding progress reporting must not add or
    reorder LLM calls. Compare a reporter-driven run against a plain run."""
    baseline_result, baseline_client = _run(None)
    reporter = RecordingReporter()
    reported_result, reported_client = _run(reporter)

    assert baseline_client.total_cost_usd == reported_client.total_cost_usd
    assert baseline_result.problem_spec.task_type == reported_result.problem_spec.task_type
    assert set(baseline_result.pipeline.train_results) == set(
        reported_result.pipeline.train_results
    )


def test_explain_model_logs_per_fold_oof_progress_tagged_as_explain():
    """explain_model's OOF refit (evaluation.oof.compute_oof, via
    _oof_permutation_importance) is the exact call that spent 45+ silent
    minutes refitting a stacking ensemble in a live run. Confirms the
    reporter receives per-fold lines tagged 'explain' -- NOT 'evaluate',
    which compute_oof also logs under for build_evaluation_report's
    calibration path; mislabeling would misattribute explainability's
    progress to the wrong dashboard stage."""
    import numpy as np
    import polars as pl

    from mlteam.agents.data_io import store_dataframe
    from mlteam.agents.explainer import explain_model
    from mlteam.agents.trainer import train_experiment
    from mlteam.contracts import BudgetEstimate, DatasetRole, ExperimentSpec, TaskType
    from mlteam.store import GuardedArtifactStore, LocalArtifactStore

    rng = np.random.default_rng(0)
    n = 150
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.3, size=n) > 0).astype(int)
    df = pl.DataFrame({"x": x, "target": y})
    root = Path(tempfile.mkdtemp()) / "store"
    store = GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")
    dataset = store_dataframe(
        store,
        df,
        role=DatasetRole.CLEANED,
        producer_agent="test",
        run_id="explain-oof-progress",
        target_column="target",
    )
    spec = ExperimentSpec(
        experiment_id="exp-model",
        template_id="lightgbm_default",
        config={"n_estimators": 15},
        budget=BudgetEstimate(wall_time_seconds=5),
    )
    trained = train_experiment(
        store,
        dataset.meta.artifact_id,
        spec,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id="explain-oof-progress",
        seed=1,
        n_folds=3,
        env_hash="test-env",
    )

    reporter = RecordingReporter()
    explain_model(
        store,
        trained.model,
        train_dataset_id=dataset.meta.artifact_id,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id="explain-oof-progress",
        primary_metric="roc_auc",
        n_folds=3,
        seed=1,
        reporter=reporter,
    )

    explain_logs = [msg for msg, _level, stage in reporter.log_calls if stage == "explain"]
    evaluate_logs = [msg for msg, _level, stage in reporter.log_calls if stage == "evaluate"]
    assert len(explain_logs) >= 3 * 2, "expected at least a start+complete line per fold"
    assert any("fold 1/3" in msg for msg in explain_logs)
    assert not evaluate_logs, "explain_model's OOF refit must not be tagged as 'evaluate'"
