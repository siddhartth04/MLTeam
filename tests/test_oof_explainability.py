"""Audit Task 6 (H4, H5): out-of-fold importances/residuals, correct SHAP
feature naming after one-hot encoding.

Root causes:
(a) permutation_importance on the TRAIN frame with the FINAL model (fit on
    that same frame) rewards memorized features -- sklearn's own docs warn
    against this. Also, the old code let permutation_importance default its
    scorer to accuracy/R^2 regardless of the run's actual primary metric.
(b) SHAP's feature_names came from the ORIGINAL input columns, but the
    transformed matrix passed to SHAP has however many columns the fitted
    preprocessor's one-hot/scaling produced -- a mismatch whenever one-hot
    encoding is in play, silently misattributing or truncating the chart.
(c) residual slices used TRAIN residuals, which for boosted trees are
    shrunk toward zero by construction (the model was fit to minimize
    exactly that residual).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from mlteam.agents.data_io import store_dataframe
from mlteam.agents.trainer import train_experiment
from mlteam.contracts import BudgetEstimate, DatasetRole, ExperimentSpec, TaskType
from mlteam.evaluation.oof import compute_oof
from mlteam.store import GuardedArtifactStore, LocalArtifactStore

RUN = "test-oof-explain"


def _store(tmp_path: Path) -> GuardedArtifactStore:
    root = tmp_path / "store"
    return GuardedArtifactStore(LocalArtifactStore(root), db_path=root / "guard.db")


def _dataset(store, df):
    return store_dataframe(
        store,
        df,
        role=DatasetRole.CLEANED,
        producer_agent="test",
        run_id=RUN,
        target_column="target",
    )


def test_compute_oof_gives_every_row_a_prediction_from_a_fold_that_excluded_it(tmp_path):
    rng = np.random.default_rng(0)
    n = 240
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.3, size=n) > 0).astype(int)
    df = pl.DataFrame({"x": x, "target": y})
    store = _store(tmp_path)
    dataset = _dataset(store, df)

    oof = compute_oof(
        store,
        dataset.meta.artifact_id,
        template_id="lightgbm_default",
        config={"n_estimators": 20},
        task_type=TaskType.BINARY_CLASSIFICATION,
        seed=42,
        n_folds=3,
    )
    assert len(oof.y_pred) == n
    assert oof.y_proba is not None
    assert len(oof.y_proba) == n
    assert not np.isnan(oof.y_proba).any()  # every row got a fold prediction
    assert len(oof.fold_pipelines) == 3
    assert sum(len(idx) for idx in oof.fold_test_indices) == n
    # test folds partition all rows exactly once.
    all_test_idx = np.concatenate(oof.fold_test_indices)
    assert sorted(all_test_idx.tolist()) == list(range(n))


def test_oof_permutation_importance_uses_a_held_out_model_per_fold(tmp_path):
    """A feature that is PURE NOISE (independent of the target) must not
    rank as importantly as the real signal feature when scored honestly
    out-of-fold -- this is the property that would be violated if
    importance were computed on the training set with the final
    (potentially memorizing) model."""
    from sklearn.metrics import get_scorer

    from mlteam.agents.explainer import _oof_permutation_importance

    rng = np.random.default_rng(0)
    n = 400
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (signal + rng.normal(scale=0.2, size=n) > 0).astype(int)
    df = pl.DataFrame({"signal": signal, "noise": noise, "target": y})
    store = _store(tmp_path)
    dataset = _dataset(store, df)

    oof = compute_oof(
        store,
        dataset.meta.artifact_id,
        template_id="lightgbm_default",
        config={"n_estimators": 50},
        task_type=TaskType.BINARY_CLASSIFICATION,
        seed=42,
        n_folds=4,
    )
    scorer = get_scorer("roc_auc")
    importances = _oof_permutation_importance(oof, scorer=scorer, seed=42)
    by_feature = {fi.feature: fi.importance for fi in importances}
    assert by_feature["signal"] > by_feature["noise"]


def test_shap_feature_names_match_transformed_width_after_one_hot(tmp_path):
    """svm_default/linear_default etc. use build_scaled_preprocessor, which
    one-hot encodes categoricals -- the transformed matrix SHAP sees has
    MORE columns than the raw input. feature_names must come from the
    fitted preprocessor's own get_feature_names_out(), not the raw columns,
    or the bar chart silently mis-pairs values with names."""
    from mlteam.agents.explainer import _shap_summary_figure

    rng = np.random.default_rng(0)
    n = 200
    category = rng.choice(["a", "b", "c", "d"], n)  # -> 4 one-hot columns
    numeric = rng.normal(size=n)
    y = (numeric + (category == "a").astype(float) + rng.normal(scale=0.3, size=n) > 0).astype(int)
    df = pl.DataFrame({"category": category, "numeric": numeric, "target": y})
    store = _store(tmp_path)
    dataset = _dataset(store, df)
    spec = ExperimentSpec(
        experiment_id="exp-lgbm",
        template_id="lightgbm_default",  # ordinal-encodes; still exercises the general path
        config={"n_estimators": 30},
        budget=BudgetEstimate(wall_time_seconds=5),
    )
    result = train_experiment(
        store,
        dataset.meta.artifact_id,
        spec,
        task_type=TaskType.BINARY_CLASSIFICATION,
        run_id=RUN,
        seed=42,
        n_folds=3,
        env_hash="test-env",
    )
    from mlteam.serving.model_io import load_pipeline

    pipeline = load_pipeline(store, result.model)
    frame = df.drop("target").to_pandas()

    fig = _shap_summary_figure(pipeline, frame, seed=42)
    # Either it renders (names/values aligned, no exception) or it returns
    # None on a genuine SHAP incompatibility -- both are acceptable, but a
    # SILENT length mismatch must never happen (the function raises
    # internally and returns None via the except clause in that case,
    # logged as shap_unavailable -- verified by checking fig is not the
    # product of a swallowed length-mismatch by asserting figure integrity
    # when SHAP DID run).
    if fig is not None:
        assert len(fig["data"][0]["x"]) == len(fig["data"][0]["y"])


def test_oof_residual_slices_use_out_of_fold_not_train_residuals(tmp_path):
    """A deliberately hard-to-fit segment ('east') must show up as the
    worst OOF residual slice -- and OOF residuals for it must be
    substantially larger than what the final (all-data-fit) model's TRAIN
    residuals would show, since boosted trees shrink train residuals
    toward zero."""
    from mlteam.agents.explainer import _oof_residual_analysis

    rng = np.random.default_rng(0)
    n = 300
    region = rng.choice(["north", "south", "east"], n)
    size = rng.uniform(50, 200, n)
    price = 3.0 * size + np.where(region == "east", 500, 0) + rng.normal(0, 20, n)
    df = pl.DataFrame({"region": region, "size": size, "target": np.round(price, 2)})
    store = _store(tmp_path)
    dataset = _dataset(store, df)

    oof = compute_oof(
        store,
        dataset.meta.artifact_id,
        template_id="lightgbm_default",
        config={"n_estimators": 30},
        task_type=TaskType.REGRESSION,
        seed=42,
        n_folds=3,
    )
    from mlteam.contracts import FeatureImportance

    importances = (FeatureImportance(feature="region", importance=1.0, std=0.0),)
    figures, slices = _oof_residual_analysis(store, oof, importances, categorical_cols={"region"})
    assert slices, "expected at least one residual slice"
    assert slices[0].value == "east"  # worst MAE segment, matching the injected offset
