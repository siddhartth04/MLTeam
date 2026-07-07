"""Explainability agent: permutation importance + PDP (+ SHAP when feasible).

Deterministic compute only (no LLM). Runs on the TRAIN dataset — the holdout
stays frozen. Produces Plotly figures (stored as JSON blobs) and a ranked
ExplainabilityReport for the report generator.

Permutation importance and residual slices are computed OUT-OF-FOLD (audit
Task 6): the final model is fit on ALL training data, so evaluating
importance/residuals on that same data with that same model rewards
memorized features and shrinks residuals toward zero for models with
enough capacity to overfit — exactly the pattern sklearn's own
documentation warns permutation_importance against. evaluation/oof.py
refits per fold (same CV split training used) and each fold's importance/
residuals are computed ONLY on that fold's held-out rows, then averaged —
genuinely out-of-sample evidence at the cost of one extra CV pass reusing
work already paid for.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import structlog

from mlteam.agents.data_io import read_dataset
from mlteam.agents.trainer import _temporal_to_epoch
from mlteam.contracts import (
    ExplainabilityReport,
    FeatureImportance,
    FigureRef,
    ImportanceMethod,
    ResidualSlice,
    TaskType,
    TrainedModel,
    is_classification,
    new_artifact,
)
from mlteam.evaluation.metrics import scoring_for
from mlteam.evaluation.oof import OOFResult, compute_oof
from mlteam.reporting.figures import (
    bar_figure,
    histogram_figure,
    line_figure,
    scatter_figure,
    store_figure,
)
from mlteam.serving.model_io import load_pipeline
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)

_TOP_PDP_FEATURES = 3
_SHAP_MAX_ROWS = 500
_MAX_RESIDUAL_SLICES = 8
_MIN_SLICE_ROWS = 5


def _oof_permutation_importance(
    oof: OOFResult, *, scorer: Any, seed: int, n_repeats: int = 5
) -> tuple[FeatureImportance, ...]:
    """Permutation importance averaged across folds, each computed ONLY on
    that fold's held-out rows with THAT fold's own fitted pipeline — never
    the final all-data model scored on data it was fit on."""
    from sklearn.inspection import permutation_importance

    feature_cols = list(oof.feature_frame.columns)
    per_fold_importances: list[np.ndarray] = []
    for fitted, test_idx in zip(oof.fold_pipelines, oof.fold_test_indices, strict=True):
        fold_x = oof.feature_frame.iloc[test_idx]
        fold_y = oof.y_true[test_idx]
        result = permutation_importance(
            fitted, fold_x, fold_y, scoring=scorer, n_repeats=n_repeats, random_state=seed, n_jobs=1
        )
        per_fold_importances.append(result.importances_mean)

    stacked = np.stack(per_fold_importances, axis=0)  # (n_folds, n_features)
    mean_importance = stacked.mean(axis=0)
    std_importance = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(
        mean_importance
    )
    ranked = sorted(
        (
            FeatureImportance(
                feature=col,
                importance=round(float(mean_importance[i]), 8),
                std=round(float(std_importance[i]), 8),
            )
            for i, col in enumerate(feature_cols)
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

    Feature names come from the fitted preprocessor's own
    ``get_feature_names_out()`` — NOT the raw input columns — because a
    one-hot/scaled preprocessor changes the column COUNT, and pairing the
    transformed values with the ORIGINAL column names silently mis-aligns
    (or truncates) the bar chart whenever one-hot encoding is in play.
    """
    try:
        import numpy as np
        import shap

        sample = frame.sample(min(len(frame), _SHAP_MAX_ROWS), random_state=seed)
        preprocess = pipeline[:-1]
        model = pipeline[-1]
        transformed = preprocess.transform(sample)
        if hasattr(preprocess, "get_feature_names_out"):
            feature_names = list(preprocess.get_feature_names_out())
        else:  # pragma: no cover - defensive; every registry preprocessor supports this
            feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):  # older SHAP: list of per-class arrays
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            # Newer SHAP: (n_samples, n_features, n_classes) -- take the
            # last (positive/highest-index) class, matching the old list-
            # based convention of values[-1].
            values = values[:, :, -1]
        mean_abs = np.abs(values).mean(axis=0)
        if len(mean_abs) != len(feature_names):
            raise ValueError(
                f"SHAP produced {len(mean_abs)} values but the preprocessor "
                f"reports {len(feature_names)} output features — refusing "
                "to render a mismatched chart"
            )
        return bar_figure(
            feature_names,
            [float(v) for v in mean_abs],
            title="SHAP mean |value| (impact on model output)",
            x_title="mean |SHAP value|",
        )
    except Exception as exc:
        _log.info("shap_unavailable", error=str(exc))
        return None


def _oof_residual_analysis(
    store: ArtifactStore,
    oof: OOFResult,
    importances: tuple[FeatureImportance, ...],
    categorical_cols: set[str],
) -> tuple[list[FigureRef], tuple[ResidualSlice, ...]]:
    """Regression error analysis on OUT-OF-FOLD predictions: residual
    distribution, residual-vs-predicted, and per-group MAE on the highest-
    importance categorical feature.

    Using OOF predictions instead of the final model's TRAIN residuals
    matters concretely for boosted trees: train residuals are shrunk toward
    zero by construction (the model was fit to minimize exactly that
    residual), making "worst slice by train MAE" close to meaningless as a
    diagnostic of real-world error.
    """
    predicted = np.asarray(oof.y_pred, dtype=float)
    actual = np.asarray(oof.y_true, dtype=float)
    residuals = actual - predicted
    frame = oof.feature_frame

    figures: list[FigureRef] = [
        FigureRef(
            kind="residual_hist",
            title="Residual distribution",
            blob_hash=store_figure(
                store,
                histogram_figure(
                    [float(r) for r in residuals],
                    title="Residual distribution (actual - predicted, out-of-fold)",
                    x_title="residual",
                ),
            ),
        ),
        FigureRef(
            kind="residual_vs_predicted",
            title="Residuals vs predicted",
            blob_hash=store_figure(
                store,
                scatter_figure(
                    [float(p) for p in predicted],
                    [float(r) for r in residuals],
                    title="Residuals vs predicted (out-of-fold)",
                    x_title="predicted",
                    y_title="residual",
                ),
            ),
        ),
    ]

    slice_feature = next(
        (fi.feature for fi in importances if fi.feature in categorical_cols), None
    )
    slices: tuple[ResidualSlice, ...] = ()
    if slice_feature is not None:
        abs_res = np.abs(residuals)
        groups: list[ResidualSlice] = []
        for value in frame[slice_feature].dropna().unique():
            mask = (frame[slice_feature] == value).to_numpy()
            n_rows = int(mask.sum())
            if n_rows < _MIN_SLICE_ROWS:
                continue
            groups.append(
                ResidualSlice(
                    feature=slice_feature,
                    value=str(value),
                    mae=round(float(abs_res[mask].mean()), 8),
                    n_rows=n_rows,
                )
            )
        groups.sort(key=lambda s: s.mae, reverse=True)
        slices = tuple(groups[:_MAX_RESIDUAL_SLICES])

    return figures, slices


def explain_model(
    store: ArtifactStore,
    model: TrainedModel,
    *,
    train_dataset_id: str,
    task_type: TaskType,
    run_id: str,
    primary_metric: str,
    n_folds: int = 5,
    seed: int = 42,
) -> ExplainabilityReport:
    """Compute importances + effect figures and store an ExplainabilityReport.

    ``task_type``/``primary_metric``/``n_folds`` drive an out-of-fold refit
    (same template, config, and CV split — via evaluation.oof.compute_oof —
    the model was trained with) used for permutation importance and (for
    regression) residual analysis — see module docstring for why train-set
    evaluation of the final model is invalid here. PDP and SHAP still use
    the final all-data-fit model directly: both describe the FITTED MODEL's
    behavior/attributions on realistic input values, not an out-of-sample
    performance estimate, so there is no equivalent leakage concern for them.
    """
    task_is_classification = is_classification(task_type)
    dataset = store.get(train_dataset_id)
    if dataset.target_column is None:
        raise ValueError("explainability needs a dataset with a target column")
    raw = read_dataset(store, dataset)
    # Categorical columns are identified from the ORIGINAL polars schema, before
    # _temporal_to_epoch turns date columns numeric — so residual slicing only
    # ever groups on genuinely categorical features.
    categorical_cols = {
        name
        for name, dtype in raw.schema.items()
        if name != dataset.target_column and dtype in (pl.String, pl.Categorical, pl.Boolean)
    }
    df = _temporal_to_epoch(raw)
    frame = df.to_pandas()
    frame.pop(dataset.target_column)  # PDP/SHAP need feature columns only

    pipeline = load_pipeline(store, model)

    from sklearn.metrics import get_scorer

    scorer = get_scorer(scoring_for(task_type)[primary_metric])
    oof = compute_oof(
        store,
        train_dataset_id,
        template_id=model.template_id,
        config=model.config,
        task_type=task_type,
        seed=seed,
        n_folds=n_folds,
    )
    importances = _oof_permutation_importance(oof, scorer=scorer, seed=seed)

    figures: list[FigureRef] = []
    perm_fig = bar_figure(
        [fi.feature for fi in importances],
        [fi.importance for fi in importances],
        title="Permutation importance (out-of-fold)",
        x_title=f"mean importance ({primary_metric} drop, out-of-fold)",
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

    # Regression error analysis: residual distribution, residual-vs-predicted,
    # and worst per-group MAE, on OUT-OF-FOLD predictions. Best-effort — never
    # blocks the report.
    residual_slices: tuple[ResidualSlice, ...] = ()
    if not task_is_classification:
        try:
            residual_figs, residual_slices = _oof_residual_analysis(
                store, oof, importances, categorical_cols
            )
            figures.extend(residual_figs)
        except Exception as exc:
            _log.info("residual_analysis_skipped", error=str(exc))

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
        residual_slices=residual_slices,
    )
    store.put(report)
    _log.info(
        "explainability_report_built",
        top_feature=importances[0].feature if importances else None,
        figures=len(figures),
        shap=shap_available,
        residual_slices=len(residual_slices),
    )
    return report
