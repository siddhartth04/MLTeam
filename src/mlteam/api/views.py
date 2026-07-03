"""Store -> JSON view helpers: artifacts, lineage graph, traces, experiments.

Pure read transforms over the store and lineage index. The holdout guard still
applies — holdout artifacts return a protected placeholder rather than content,
so the dashboard can show that the holdout exists without ever reading it.
"""

from __future__ import annotations

from typing import Any

from mlteam.contracts import (
    CritiqueReport,
    EvaluationReport,
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
    store: GuardedArtifactStore, *, run_id: str | None = None, artifact_type: str | None = None
) -> list[dict[str, Any]]:
    ids = store.list_ids(artifact_type)
    records = [store._inner.lineage.get_record(aid) for aid in ids]
    out = []
    for record in records:
        if record is None:
            continue
        if run_id is not None and record.run_id != run_id:
            continue
        out.append(artifact_summary(record))
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


def lineage_graph(store: GuardedArtifactStore, run_id: str | None = None) -> dict[str, Any]:
    """Nodes + edges for the artifact-lineage explorer."""
    lineage = store._inner.lineage
    ids = store.list_ids()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for aid in ids:
        record = lineage.get_record(aid)
        if record is None or (run_id is not None and record.run_id != run_id):
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


def experiments_view(store: GuardedArtifactStore, run_id: str) -> list[dict[str, Any]]:
    """Leaderboard rows: every TrainedModel for a run + its CV metrics."""
    rows: list[dict[str, Any]] = []
    for aid in store.list_ids(TrainedModel.artifact_type()):
        record = store._inner.lineage.get_record(aid)
        if record is None or record.run_id != run_id:
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
                "seed": model.seed,
                "metrics": metrics,
                "is_baseline": model.template_id == "dummy_baseline",
            }
        )
    rows.sort(key=lambda r: r["experiment_id"])
    return rows


def _evaluation_for_model(
    store: GuardedArtifactStore, model_id: str, run_id: str
) -> EvaluationReport | None:
    for aid in store.list_ids(EvaluationReport.artifact_type()):
        record = store._inner.lineage.get_record(aid)
        if record is None or record.run_id != run_id:
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
    for aid in store.list_ids(CritiqueReport.artifact_type()):
        record = store._inner.lineage.get_record(aid)
        if record is None or record.run_id != run_id:
            continue
        if record.created_at >= latest_at:
            latest_at = record.created_at
            latest = store.get(aid)
    return latest.payload_dict() if latest is not None else None


def problem_spec_view(store: GuardedArtifactStore, run_id: str) -> dict[str, Any] | None:
    confirmed: dict[str, Any] | None = None
    for aid in store.list_ids(ProblemSpec.artifact_type()):
        record = store._inner.lineage.get_record(aid)
        if record is None or record.run_id != run_id:
            continue
        spec: ProblemSpec = store.get(aid)
        payload = spec.payload_dict()
        if spec.human_confirmed:
            return payload
        confirmed = confirmed or payload
    return confirmed


def explainability_view(store: GuardedArtifactStore, run_id: str) -> dict[str, Any] | None:
    for aid in store.list_ids(ExplainabilityReport.artifact_type()):
        record = store._inner.lineage.get_record(aid)
        if record is None or record.run_id != run_id:
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


def final_report_view(store: GuardedArtifactStore, artifact_id: str) -> FinalReport | None:
    try:
        return store.get(artifact_id)
    except Exception:
        return None
