"""Store -> JSON view helpers: artifacts, lineage graph, traces, experiments.

Pure read transforms over the store and lineage index. The holdout guard still
applies — holdout artifacts return a protected placeholder rather than content,
so the dashboard can show that the holdout exists without ever reading it.
"""

from __future__ import annotations

from typing import Any

from mlteam.contracts import (
    CritiqueReport,
    DeploymentBundle,
    EvaluationReport,
    ExperimentOutcome,
    ExperimentPlan,
    ExplainabilityReport,
    FinalReport,
    ProblemSpec,
    TrainedModel,
)
from mlteam.store import GuardedArtifactStore, HoldoutAccessError
from mlteam.store.lineage import LineageRecord


def artifact_summary(record: LineageRecord) -> dict[str, Any]:
    return {
        "id": record.artifact_id,
        "short_id": record.artifact_id[:12],
        "type": record.artifact_type,
        "producer": record.producer_agent,
        "run_id": record.run_id,
        "config_hash": record.config_hash,
        "created_at": record.created_at,
    }


def list_artifacts(
    store: GuardedArtifactStore, *, run_id: str, artifact_type: str | None = None
) -> list[dict[str, Any]]:
    """Artifacts scoped to a single pipeline run.

    ``run_id`` is required (not Optional) on purpose: a ``None`` here used to
    mean "no filter", so an API route with an unresolved pipeline run id (a
    run still in flight, or an unknown ui id) would silently return every
    artifact in the store across every run ever recorded. Callers that
    genuinely want a whole-store listing must use ``list_all_artifacts``, so
    that intent is explicit at the call site instead of an accidental default.
    """
    scoped = store.artifact_ids_for_run(run_id)
    ids = store.list_ids(artifact_type)
    records = [store._inner.lineage.get_record(aid) for aid in ids]
    out = []
    for record in records:
        if record is None:
            continue
        if not store.is_in_run(record.artifact_id, run_id, scoped):
            continue
        out.append(artifact_summary(record))
    out.sort(key=lambda a: a["created_at"])
    return out


def list_all_artifacts(
    store: GuardedArtifactStore, *, artifact_type: str | None = None
) -> list[dict[str, Any]]:
    """Every artifact in the store, unscoped by run. Not wired to any API
    route -- exists so a deliberate whole-store listing is a distinct,
    explicitly-named call rather than the (former) default behavior of
    ``list_artifacts`` when it's handed a ``None`` run id."""
    ids = store.list_ids(artifact_type)
    records = [store._inner.lineage.get_record(aid) for aid in ids]
    out = [artifact_summary(record) for record in records if record is not None]
    out.sort(key=lambda a: a["created_at"])
    return out


def artifact_detail(store: GuardedArtifactStore, artifact_id: str) -> dict[str, Any] | None:
    record = store._inner.lineage.get_record(artifact_id)
    if record is None:
        return None
    summary = artifact_summary(record)
    lineage = store._inner.lineage
    summary["parents"] = list(lineage.parents(artifact_id))
    summary["children"] = list(lineage.children(artifact_id))
    summary["protected"] = store.is_protected(artifact_id)
    if summary["protected"]:
        summary["payload"] = {"_protected": "holdout artifact — content withheld until reporting"}
    else:
        try:
            artifact = store.get(artifact_id)
            summary["payload"] = artifact.payload_dict()
        except HoldoutAccessError:
            summary["payload"] = {"_protected": "holdout artifact — content withheld"}
    return summary


def lineage_graph(store: GuardedArtifactStore, run_id: str) -> dict[str, Any]:
    """Nodes + edges for the artifact-lineage explorer, scoped to one run.

    ``run_id`` is required for the same reason as ``list_artifacts``: a
    ``None`` default previously meant "show the whole store", which silently
    mixed unrelated runs' artifacts into a single graph whenever the caller's
    run id hadn't resolved yet.
    """
    lineage = store._inner.lineage
    scoped = store.artifact_ids_for_run(run_id)
    ids = store.list_ids()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for aid in ids:
        record = lineage.get_record(aid)
        if record is None:
            continue
        if not store.is_in_run(aid, run_id, scoped):
            continue
        nodes.append(
            {
                "id": aid,
                "short_id": aid[:10],
                "type": record.artifact_type,
                "producer": record.producer_agent,
                "protected": store.is_protected(aid),
            }
        )
        for parent in lineage.parents(aid):
            edges.append({"source": parent, "target": aid})
    node_ids = {n["id"] for n in nodes}
    edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]
    return {"nodes": nodes, "edges": edges}


def _is_baseline_lookup(store: GuardedArtifactStore, run_id: str) -> dict[str, bool]:
    """experiment_id -> is_baseline, unioned across every plan revision.

    The plan is the actual source of truth for "is this a baseline" (the LLM
    marks dummy_baseline AND the incumbent template, e.g. lightgbm_default,
    is_baseline=true — see experiment_planner.py). template_id ==
    "dummy_baseline" alone under-counts (misses the marked incumbent) and,
    separately, screening experiments (screen-dummy, screen-linear, ...) run
    BEFORE any plan exists and so never appear in one — those correctly fall
    through to the template_id heuristic below.
    """
    scoped = store.artifact_ids_for_run(run_id)
    lookup: dict[str, bool] = {}
    for aid in store.list_ids(ExperimentPlan.artifact_type()):
        if not store.is_in_run(aid, run_id, scoped):
            continue
        plan: ExperimentPlan = store.get(aid)
        for exp in plan.experiments:
            lookup[exp.experiment_id] = lookup.get(exp.experiment_id, False) or exp.is_baseline
    return lookup


def _resolve_is_baseline(
    lookup: dict[str, bool], experiment_id: str, template_id: str
) -> bool:
    if experiment_id in lookup:
        return lookup[experiment_id]
    return template_id == "dummy_baseline"


def experiments_view(store: GuardedArtifactStore, run_id: str) -> list[dict[str, Any]]:
    """Leaderboard rows: every experiment attempt for a run, whatever its fate.

    Reads ExperimentOutcome artifacts — persisted for every planned experiment
    regardless of whether it completed, failed, or was skipped — so a crashed
    template renders as a visible failure with its error, not as a blank row
    indistinguishable from "never ran". Falls back to the legacy TrainedModel
    join (winner-only metrics) for runs recorded before outcomes existed.
    """
    rows: list[dict[str, Any]] = []
    scoped = store.artifact_ids_for_run(run_id)
    baseline_lookup = _is_baseline_lookup(store, run_id)
    for aid in store.list_ids(ExperimentOutcome.artifact_type()):
        record = store._inner.lineage.get_record(aid)
        if record is None or not store.is_in_run(aid, run_id, scoped):
            continue
        outcome: ExperimentOutcome = store.get(aid)
        rows.append(
            {
                "artifact_id": aid,
                "experiment_id": outcome.experiment_id,
                "template_id": outcome.template_id,
                "config": outcome.config,
                "state": outcome.state.value,
                "metrics": {m.name: {"mean": m.mean, "std": m.std} for m in outcome.cv_metrics},
                "wall_time_seconds": outcome.wall_time_seconds,
                "error": outcome.error,
                "is_baseline": _resolve_is_baseline(
                    baseline_lookup, outcome.experiment_id, outcome.template_id
                ),
            }
        )
    if rows:
        rows.sort(key=lambda r: r["experiment_id"])
        return rows
    return _legacy_experiments_view(store, run_id)


def _legacy_experiments_view(store: GuardedArtifactStore, run_id: str) -> list[dict[str, Any]]:
    """Pre-ExperimentOutcome fallback: winner-only metrics via TrainedModel."""
    rows: list[dict[str, Any]] = []
    scoped = store.artifact_ids_for_run(run_id)
    baseline_lookup = _is_baseline_lookup(store, run_id)
    for aid in store.list_ids(TrainedModel.artifact_type()):
        record = store._inner.lineage.get_record(aid)
        if record is None or not store.is_in_run(aid, run_id, scoped):
            continue
        model: TrainedModel = store.get(aid)
        report = _evaluation_for_model(store, aid, run_id)
        metrics = (
            {m.name: {"mean": m.mean, "std": m.std} for m in report.cv_metrics}
            if report is not None
            else {}
        )
        rows.append(
            {
                "artifact_id": aid,
                "experiment_id": model.experiment_id,
                "template_id": model.template_id,
                "config": model.config,
                "state": "completed",
                "metrics": metrics,
                "wall_time_seconds": None,
                "error": None,
                "is_baseline": _resolve_is_baseline(
                    baseline_lookup, model.experiment_id, model.template_id
                ),
            }
        )
    rows.sort(key=lambda r: r["experiment_id"])
    return rows


def _evaluation_for_model(
    store: GuardedArtifactStore, model_id: str, run_id: str
) -> EvaluationReport | None:
    scoped = store.artifact_ids_for_run(run_id)
    for aid in store.list_ids(EvaluationReport.artifact_type()):
        if not store.is_in_run(aid, run_id, scoped):
            continue
        report: EvaluationReport = store.get(aid)
        if report.model_ref.artifact_id == model_id:
            return report
    return None


def evaluation_view(store: GuardedArtifactStore, artifact_id: str) -> dict[str, Any] | None:
    try:
        report: EvaluationReport = store.get(artifact_id)
    except Exception:
        return None
    return report.payload_dict()


def critique_view(store: GuardedArtifactStore, run_id: str) -> dict[str, Any] | None:
    latest: CritiqueReport | None = None
    latest_at = ""
    scoped = store.artifact_ids_for_run(run_id)
    for aid in store.list_ids(CritiqueReport.artifact_type()):
        record = store._inner.lineage.get_record(aid)
        if record is None or not store.is_in_run(aid, run_id, scoped):
            continue
        if record.created_at >= latest_at:
            latest_at = record.created_at
            latest = store.get(aid)
    return latest.payload_dict() if latest is not None else None


def problem_spec_view(store: GuardedArtifactStore, run_id: str) -> dict[str, Any] | None:
    confirmed: dict[str, Any] | None = None
    scoped = store.artifact_ids_for_run(run_id)
    for aid in store.list_ids(ProblemSpec.artifact_type()):
        if not store.is_in_run(aid, run_id, scoped):
            continue
        spec: ProblemSpec = store.get(aid)
        payload = spec.payload_dict()
        if spec.human_confirmed:
            return payload
        confirmed = confirmed or payload
    return confirmed


def explainability_view(store: GuardedArtifactStore, run_id: str) -> dict[str, Any] | None:
    scoped = store.artifact_ids_for_run(run_id)
    for aid in store.list_ids(ExplainabilityReport.artifact_type()):
        if not store.is_in_run(aid, run_id, scoped):
            continue
        report: ExplainabilityReport = store.get(aid)
        return {
            "importances": [
                {"feature": fi.feature, "importance": fi.importance, "std": fi.std}
                for fi in report.importances
            ],
            "shap_available": report.shap_available,
            "figures": [
                {"kind": f.kind, "title": f.title, "blob_hash": f.blob_hash, "feature": f.feature}
                for f in report.figures
            ],
        }
    return None


def deployment_bundle_for_run(
    store: GuardedArtifactStore, run_id: str
) -> DeploymentBundle | None:
    """The run's signed-off DeploymentBundle, if Gate 3 has completed.

    audit M8: drift monitoring needed a way to find the bundle a drift
    check should run against; every other run-scoped lookup in this module
    already follows this exact membership-table pattern (Task 1's fix).
    """
    scoped = store.artifact_ids_for_run(run_id)
    latest: DeploymentBundle | None = None
    for aid in store.list_ids(DeploymentBundle.artifact_type()):
        if not store.is_in_run(aid, run_id, scoped):
            continue
        latest = store.get(aid)
    return latest


def final_report_view(store: GuardedArtifactStore, artifact_id: str) -> FinalReport | None:
    try:
        return store.get(artifact_id)
    except Exception:
        return None
