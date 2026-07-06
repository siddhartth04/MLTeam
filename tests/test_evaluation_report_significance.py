"""Audit Task 2 (H1): build_evaluation_report uses the real paired
Nadeau-Bengio test when fold pairing + n_rows are available, and falls back
to the documented heuristic (with significance_method recorded) otherwise.
"""

from __future__ import annotations

from mlteam.contracts import DataProfile, MetricSummary, TrainedModel, new_artifact
from mlteam.evaluation.report import build_evaluation_report
from mlteam.evaluation.results import TrainResult
from mlteam.store import GuardedArtifactStore, LocalArtifactStore

RUN = "test-significance"


def _metric(name: str, fold_values: tuple[float, ...]) -> MetricSummary:
    mean = sum(fold_values) / len(fold_values)
    n = len(fold_values)
    var = sum((v - mean) ** 2 for v in fold_values) / (n - 1) if n > 1 else 0.0
    return MetricSummary(name=name, fold_values=fold_values, mean=mean, std=var**0.5)


def _model(experiment_id: str) -> TrainedModel:
    return new_artifact(
        TrainedModel,
        producer_agent="test",
        run_id=RUN,
        experiment_id=experiment_id,
        template_id="lightgbm_default",
        pipeline_blob="a" * 64,
        config={},
        seed=42,
        env_hash="test-env",
        train_data_ref={"artifact_id": "ds-1", "artifact_type": "DatasetVersion"},
    )


def _profile() -> DataProfile:
    return new_artifact(
        DataProfile,
        producer_agent="test",
        run_id=RUN,
        dataset_ref={"artifact_id": "ds-1", "artifact_type": "DatasetVersion"},
        columns=(),
    )


def _store(tmp_path) -> GuardedArtifactStore:
    root = tmp_path / "store"
    return GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")


def test_paired_folds_and_n_rows_produce_a_real_p_value(tmp_path):
    store = _store(tmp_path)
    model = TrainResult(
        model=_model("exp-model"),
        metrics={"roc_auc": _metric("roc_auc", (0.56, 0.71, 0.84))},
    )
    dummy = TrainResult(
        model=_model("exp-dummy"),
        metrics={"roc_auc": _metric("roc_auc", (0.50, 0.65, 0.80))},
    )
    profile = _profile()
    store.put(model.model)
    store.put(dummy.model)
    store.put(profile)
    report = build_evaluation_report(
        store,
        run_id=RUN,
        model_result=model,
        dummy_result=dummy,
        profile=profile,
        primary_metric="roc_auc",
        n_rows=300,
    )
    delta = next(d for d in report.baseline_deltas if d.metric == "roc_auc")
    assert delta.significance_method == "nadeau_bengio_paired_t"
    assert delta.p_value is not None
    assert delta.p_value < 0.05
    assert delta.significant is True


def test_mismatched_fold_counts_fall_back_to_heuristic(tmp_path):
    store = _store(tmp_path)
    model = TrainResult(
        model=_model("exp-model"),
        metrics={"roc_auc": _metric("roc_auc", (0.85, 0.86, 0.87, 0.88, 0.89))},
    )
    dummy = TrainResult(
        model=_model("exp-dummy"),
        metrics={"roc_auc": _metric("roc_auc", (0.50, 0.51))},
    )
    profile = _profile()
    store.put(model.model)
    store.put(dummy.model)
    store.put(profile)
    report = build_evaluation_report(
        store,
        run_id=RUN,
        model_result=model,
        dummy_result=dummy,
        profile=profile,
        primary_metric="roc_auc",
        n_rows=300,
    )
    delta = next(d for d in report.baseline_deltas if d.metric == "roc_auc")
    assert delta.significance_method == "heuristic_2sigma"
    assert delta.p_value is None


def test_omitted_n_rows_falls_back_to_heuristic(tmp_path):
    store = _store(tmp_path)
    model = TrainResult(
        model=_model("exp-model"),
        metrics={"roc_auc": _metric("roc_auc", (0.85, 0.86, 0.87))},
    )
    dummy = TrainResult(
        model=_model("exp-dummy"),
        metrics={"roc_auc": _metric("roc_auc", (0.50, 0.51, 0.52))},
    )
    profile = _profile()
    store.put(model.model)
    store.put(dummy.model)
    store.put(profile)
    report = build_evaluation_report(
        store,
        run_id=RUN,
        model_result=model,
        dummy_result=dummy,
        profile=profile,
        primary_metric="roc_auc",
        # n_rows omitted entirely
    )
    delta = next(d for d in report.baseline_deltas if d.metric == "roc_auc")
    assert delta.significance_method == "heuristic_2sigma"
    assert delta.p_value is None
