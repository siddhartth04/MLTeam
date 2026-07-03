"""Explainability agent: permutation importance + PDP (+ SHAP when feasible).

Deterministic compute only (no LLM). Runs on the TRAIN dataset — the holdout
stays frozen. Produces Plotly figures (stored as JSON blobs) and a ranked
ExplainabilityReport for the report generator.
"""

from __future__ import annotations

from typing import Any

import structlog

from mlteam.agents.data_io import read_dataset
from mlteam.agents.trainer import _temporal_to_epoch
from mlteam.contracts import (
    ExplainabilityReport,
    FeatureImportance,
    FigureRef,
    ImportanceMethod,
    TrainedModel,
    new_artifact,
)
from mlteam.reporting.figures import bar_figure, line_figure, store_figure
from mlteam.serving.model_io import load_pipeline
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)

_TOP_PDP_FEATURES = 3
_SHAP_MAX_ROWS = 500


def _permutation_importance(
    pipeline: Any, frame: Any, target: Any, *, seed: int
) -> tuple[FeatureImportance, ...]:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline, frame, target, n_repeats=5, random_state=seed, n_jobs=1
    )
    ranked = sorted(
        (
            FeatureImportance(
                feature=col,
                importance=round(float(result.importances_mean[i]), 8),
                std=round(float(result.importances_std[i]), 8),
            )
            for i, col in enumerate(frame.columns)
        ),
        key=lambda fi: fi.importance,
        reverse=True,
    )
    return tuple(ranked)


def _partial_dependence_figure(
    pipeline: Any, frame: Any, feature: str, *, task_is_classification: bool
) -> Any:
    from sklearn.inspection import partial_dependence

    kwargs = {"kind": "average"}
    if task_is_classification and hasattr(pipeline, "predict_proba"):
        kwargs["response_method"] = "predict_proba"
    pd_result = partial_dependence(pipeline, frame, [feature], **kwargs)
    grid = [float(v) for v in pd_result["grid_values"][0]]
    average = [float(v) for v in pd_result["average"][0]]
    return line_figure(grid, average, title=f"Partial dependence: {feature}", x_title=feature)


def _shap_summary_figure(pipeline: Any, frame: Any, *, seed: int) -> Any:
    """Best-effort SHAP on the final estimator over preprocessed features.

    SHAP is optional-by-behavior: any failure returns None and is logged, so
    explainability never blocks on it.
    """
    try:
        import numpy as np
        import shap

        sample = frame.sample(min(len(frame), _SHAP_MAX_ROWS), random_state=seed)
        preprocess = pipeline[:-1]
        model = pipeline[-1]
        transformed = preprocess.transform(sample)
        feature_names = list(sample.columns)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):  # binary classifier -> per-class
            values = values[-1]
        mean_abs = np.abs(values).mean(axis=0)
        return bar_figure(
            feature_names,
            [float(v) for v in mean_abs],
            title="SHAP mean |value| (impact on model output)",
            x_title="mean |SHAP value|",
        )
    except Exception as exc:
        _log.info("shap_unavailable", error=str(exc))
        return None


def explain_model(
    store: ArtifactStore,
    model: TrainedModel,
    *,
    train_dataset_id: str,
    task_is_classification: bool,
    run_id: str,
    seed: int = 42,
) -> ExplainabilityReport:
    """Compute importances + effect figures and store an ExplainabilityReport."""
    dataset = store.get(train_dataset_id)
    if dataset.target_column is None:
        raise ValueError("explainability needs a dataset with a target column")
    df = _temporal_to_epoch(read_dataset(store, dataset))
    frame = df.to_pandas()
    target = frame.pop(dataset.target_column)

    pipeline = load_pipeline(store, model)
    importances = _permutation_importance(pipeline, frame, target, seed=seed)

    figures: list[FigureRef] = []
    perm_fig = bar_figure(
        [fi.feature for fi in importances],
        [fi.importance for fi in importances],
        title="Permutation importance",
        x_title="mean importance (metric drop)",
    )
    figures.append(
        FigureRef(
            kind="permutation_importance",
            title="Permutation importance",
            blob_hash=store_figure(store, perm_fig),
        )
    )

    for fi in importances[:_TOP_PDP_FEATURES]:
        try:
            pdp_fig = _partial_dependence_figure(
                pipeline, frame, fi.feature, task_is_classification=task_is_classification
            )
        except Exception as exc:  # PDP can fail on some feature types
            _log.info("pdp_skipped", feature=fi.feature, error=str(exc))
            continue
        figures.append(
            FigureRef(
                kind="partial_dependence",
                title=f"Partial dependence: {fi.feature}",
                blob_hash=store_figure(store, pdp_fig),
                feature=fi.feature,
            )
        )

    shap_fig = _shap_summary_figure(pipeline, frame, seed=seed)
    shap_available = shap_fig is not None
    if shap_fig is not None:
        figures.append(
            FigureRef(
                kind="shap_summary",
                title="SHAP summary",
                blob_hash=store_figure(store, shap_fig),
            )
        )

    report = new_artifact(
        ExplainabilityReport,
        producer_agent="explainer",
        run_id=run_id,
        parent_ids=(model.meta.artifact_id, train_dataset_id),
        model_ref=model.ref(),
        method=ImportanceMethod.PERMUTATION,
        importances=importances,
        shap_available=shap_available,
        figures=tuple(figures),
    )
    store.put(report)
    _log.info(
        "explainability_report_built",
        top_feature=importances[0].feature if importances else None,
        figures=len(figures),
        shap=shap_available,
    )
    return report
