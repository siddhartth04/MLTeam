"""Prefect flow wrapping the deterministic pipeline as cached tasks.

Every task takes/returns plain values (artifact ids, small dicts) by
convention — Prefect persists results with its default (pickle) serializer.
Cache keys exclude run_id and hash file CONTENT for path inputs — identical
data + config = cache hit, mirroring the store's content-addressed dedup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.context import TaskRunContext

from mlteam.agents.cleaner import execute_cleaning, propose_cleaning_ops
from mlteam.agents.ingest import ingest_file
from mlteam.agents.profiler import profile_dataset
from mlteam.agents.splitter import plan_split
from mlteam.agents.trainer import TrainResult, compute_env_hash, train_experiment
from mlteam.contracts import (
    ExperimentSpec,
    TaskType,
    content_hash,
    file_content_hash,
)
from mlteam.evaluation.metrics import summarize_folds
from mlteam.evaluation.report import build_evaluation_report
from mlteam.orchestration.pipeline import default_experiment_specs, new_run_id
from mlteam.store import GuardedArtifactStore, open_store


def _stable_cache_key(context: TaskRunContext, parameters: dict[str, Any]) -> str:
    """Cache on inputs minus run identity; file params hash by content."""
    payload = {k: v for k, v in parameters.items() if k != "run_id"}
    path = payload.get("data_path")
    if path is not None:
        payload["data_content"] = file_content_hash(Path(path))
    return content_hash({"task": context.task.name, "inputs": payload})


def _store(store_root: str) -> GuardedArtifactStore:
    return open_store(Path(store_root))


@task(cache_key_fn=_stable_cache_key, persist_result=True)
def ingest_step(store_root: str, data_path: str, target_column: str, run_id: str) -> str:
    dataset = ingest_file(
        _store(store_root), Path(data_path), target_column=target_column, run_id=run_id
    )
    return dataset.meta.artifact_id


@task(cache_key_fn=_stable_cache_key, persist_result=True)
def profile_step(store_root: str, dataset_id: str, seed: int, run_id: str) -> str:
    profile = profile_dataset(_store(store_root), dataset_id, run_id=run_id, seed=seed)
    return profile.meta.artifact_id


@task(cache_key_fn=_stable_cache_key, persist_result=True)
def split_step(
    store_root: str,
    dataset_id: str,
    task_type: str,
    seed: int,
    holdout_fraction: float,
    n_folds: int,
    run_id: str,
) -> dict[str, str]:
    plan = plan_split(
        _store(store_root),
        dataset_id,
        task_type=TaskType(task_type),
        run_id=run_id,
        seed=seed,
        holdout_fraction=holdout_fraction,
        n_folds=n_folds,
    )
    return {"plan_id": plan.meta.artifact_id, "train_id": plan.train_ref.artifact_id}


@task(cache_key_fn=_stable_cache_key, persist_result=True)
def clean_step(store_root: str, dataset_id: str, profile_id: str, run_id: str) -> dict[str, str]:
    store = _store(store_root)
    profile = store.get(profile_id)
    cleaned, manifest = execute_cleaning(
        store, dataset_id, propose_cleaning_ops(profile), run_id=run_id
    )
    return {"cleaned_id": cleaned.meta.artifact_id, "manifest_id": manifest.meta.artifact_id}


@task(cache_key_fn=_stable_cache_key, persist_result=True)
def train_step(
    store_root: str,
    dataset_id: str,
    spec: dict[str, Any],
    task_type: str,
    seed: int,
    n_folds: int,
    mlflow_enabled: bool,
    run_id: str,
) -> dict[str, Any]:
    result = train_experiment(
        _store(store_root),
        dataset_id,
        ExperimentSpec.model_validate(spec),
        task_type=TaskType(task_type),
        run_id=run_id,
        seed=seed,
        n_folds=n_folds,
        env_hash=compute_env_hash(),
        mlflow_enabled=mlflow_enabled,
    )
    return {
        "model_id": result.model.meta.artifact_id,
        "metrics": {name: list(s.fold_values) for name, s in result.metrics.items()},
    }


def _rebuild_result(store: GuardedArtifactStore, entry: dict[str, Any]) -> TrainResult:
    model = store.get(entry["model_id"])
    metrics = {name: summarize_folds(name, folds) for name, folds in entry["metrics"].items()}
    return TrainResult(model=model, metrics=metrics)


@task(cache_key_fn=_stable_cache_key, persist_result=True)
def evaluate_step(
    store_root: str,
    model_entry: dict[str, Any],
    dummy_entry: dict[str, Any],
    profile_id: str,
    primary_metric: str,
    run_id: str,
) -> str:
    store = _store(store_root)
    report = build_evaluation_report(
        store,
        run_id=run_id,
        model_result=_rebuild_result(store, model_entry),
        dummy_result=_rebuild_result(store, dummy_entry),
        profile=store.get(profile_id),
        primary_metric=primary_metric,
    )
    return report.meta.artifact_id


def _best_entry(
    entries: dict[str, dict[str, Any]], primary_metric: str, dummy_key: str
) -> tuple[str, dict[str, Any]]:
    """Entries are keyed by experiment_id — templates may repeat across specs."""
    from mlteam.evaluation.metrics import better_than

    def mean(entry: dict[str, Any]) -> float:
        folds = entry["metrics"][primary_metric]
        return sum(folds) / len(folds)

    candidates = {eid: e for eid, e in entries.items() if eid != dummy_key}
    best_id = next(iter(candidates))
    for experiment_id, entry in candidates.items():
        if better_than(primary_metric, mean(entry), mean(candidates[best_id])):
            best_id = experiment_id
    return best_id, candidates[best_id]


@flow(name="tabular-baseline-pipeline")
def tabular_baseline_flow(
    data_path: str,
    target_column: str,
    task_type: str,
    primary_metric: str,
    store_root: str = "./artifact_store",
    seed: int = 42,
    holdout_fraction: float = 0.2,
    n_folds: int = 5,
    include_tabpfn: bool = True,
    mlflow_enabled: bool = False,
) -> dict[str, Any]:
    """Phase 1 flow: Prefect owns retries/caching; agents stay pure functions."""
    run_id = new_run_id()
    raw_id = ingest_step(store_root, data_path, target_column, run_id)
    raw_profile_id = profile_step(store_root, raw_id, seed, run_id)
    split_info = split_step(store_root, raw_id, task_type, seed, holdout_fraction, n_folds, run_id)
    train_profile_id = profile_step(store_root, split_info["train_id"], seed, run_id)
    clean_info = clean_step(store_root, split_info["train_id"], train_profile_id, run_id)

    entries: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    specs = default_experiment_specs(include_tabpfn=include_tabpfn)
    dummy_key = next(s.experiment_id for s in specs if s.template_id == "dummy_baseline")
    for spec in specs:
        state = train_step(
            store_root,
            clean_info["cleaned_id"],
            spec.model_dump(mode="json"),
            task_type,
            seed,
            n_folds,
            mlflow_enabled,
            run_id,
            return_state=True,
        )
        if state.is_completed():
            entries[spec.experiment_id] = state.result()
        elif spec.template_id == "dummy_baseline":
            raise RuntimeError("dummy baseline failed; evaluation is meaningless without it")
        else:
            skipped[spec.experiment_id] = str(state.message)

    best_id, best_entry = _best_entry(entries, primary_metric, dummy_key)
    report_id = evaluate_step(
        store_root,
        best_entry,
        entries[dummy_key],
        train_profile_id,
        primary_metric,
        run_id,
    )
    return {
        "run_id": run_id,
        "raw_dataset_id": raw_id,
        "raw_profile_id": raw_profile_id,
        "split": split_info,
        "cleaned": clean_info,
        "best_template_id": best_id,
        "skipped": skipped,
        "evaluation_report_id": report_id,
    }
