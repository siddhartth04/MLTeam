"""Registry of vetted pipeline templates.

Each template builds ONE sklearn Pipeline (preprocessing + model as a single
object) so preprocessing is always fit inside CV folds and the serialized
artifact is the complete inference path. Agents select templates by id and
fill in config — they never write model code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mlteam.contracts import TaskType

_CLASSIFICATION = frozenset(
    {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        # Labeled anomaly detection is imbalanced binary classification to
        # every classifier template; dedicated detectors below add the
        # unsupervised-fit alternative.
        TaskType.ANOMALY_DETECTION,
    }
)
# Forecasting reuses the regressor branch of every template: after temporal
# splitting + lag features it is direct supervised regression.
_REGRESSION_LIKE = frozenset({TaskType.REGRESSION, TaskType.FORECASTING})
_ALL_TASKS = _CLASSIFICATION | _REGRESSION_LIKE
_ANOMALY_ONLY = frozenset({TaskType.ANOMALY_DETECTION})

# TabPFN operating envelope (v2, CPU): beyond this it is the wrong tool.
TABPFN_MAX_ROWS = 10_000
TABPFN_MAX_FEATURES = 500


@dataclass(frozen=True)
class ColumnRoles:
    """Feature columns split by how the preprocessor must treat them."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


@dataclass(frozen=True)
class PipelineTemplate:
    template_id: str
    description: str
    task_types: frozenset[TaskType]
    is_baseline: bool
    build: Callable[[TaskType, ColumnRoles, Mapping[str, Any], int], Any]

    def supports(self, task_type: TaskType) -> bool:
        return task_type in self.task_types


def build_preprocessor(roles: ColumnRoles) -> Any:
    """Impute + ordinal-encode inside the pipeline (leakage-safe by design).

    Right for tree ensembles (LightGBM/XGBoost/RF/...), which are scale-free
    and split on ordinal codes without harm.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(roles.numeric)),
            ("categorical", categorical, list(roles.categorical)),
        ],
        remainder="drop",
    )


def build_scaled_preprocessor(roles: ColumnRoles) -> Any:
    """Impute + standardize + one-hot inside the pipeline.

    Required by distance/margin/gradient models (linear, SVM, KNN, MLP,
    GaussianNB): ordinal codes would impose fake ordering and unscaled
    features would dominate distances. ``max_categories`` caps blow-up on
    high-cardinality columns; unseen levels fold into the infrequent bucket.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=20,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(roles.numeric)),
            ("categorical", categorical, list(roles.categorical)),
        ],
        remainder="drop",
    )


def build_catboost_preprocessor(roles: ColumnRoles) -> Any:
    """Impute only — categoricals stay as raw strings for CatBoost's own
    ordered-target-statistic encoding, its actual competitive advantage over
    ordinal/one-hot tree ensembles on categorical-heavy data (audit L2:
    catboost_default previously reused the generic ordinal preprocessor,
    discarding that advantage entirely).

    Column order is fixed (numeric block, then categorical block, matching
    ``roles``), so the categorical block's positions are deterministic —
    ``_build_catboost`` uses that to build ``cat_features`` indices.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    # A constant sentinel via pandas .fillna(), NOT SimpleImputer: CatBoost's
    # native categorical path requires cat_features columns to be entirely
    # non-null strings (raises TypeError on a raw None/NaN reaching
    # catboost.Pool -- verified directly). SimpleImputer's default
    # missing_values=np.nan does not match a Python None sitting in an
    # object-dtype column (as polars null -> pandas object commonly
    # produces), so mixed None/NaN representations in the same column can
    # silently pass an unfilled None straight through -- reproduced directly
    # against sklearn 1.x. pandas .fillna() recognizes both uniformly
    # regardless of dtype, so it's used here instead. The "__missing__"
    # token also lets CatBoost treat "was missing" as its own informative
    # category, standard practice for CatBoost specifically (unlike the
    # ordinal-encoded templates, where a missing indicator is not exposed).
    categorical = Pipeline(
        [
            (
                "fill_missing",
                FunctionTransformer(lambda df: df.fillna("__missing__").astype(str)),
            )
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(roles.numeric)),
            ("categorical", categorical, list(roles.categorical)),
        ],
        remainder="drop",
    )


def _build_dummy(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.pipeline import Pipeline

    if task_type in _CLASSIFICATION:
        model = DummyClassifier(strategy="prior", random_state=seed)
    else:
        model = DummyRegressor(strategy="mean")
    return Pipeline([("preprocess", build_preprocessor(roles)), ("model", model)])


def _build_lightgbm(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.pipeline import Pipeline

    params: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "random_state": seed,
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": 1,
        "verbosity": -1,
        **dict(config),
    }
    model_cls = LGBMClassifier if task_type in _CLASSIFICATION else LGBMRegressor
    return Pipeline([("preprocess", build_preprocessor(roles)), ("model", model_cls(**params))])


def _build_tabpfn(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.pipeline import Pipeline
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    params: dict[str, Any] = {"device": "cpu", "random_state": seed, **dict(config)}
    model_cls = TabPFNClassifier if task_type in _CLASSIFICATION else TabPFNRegressor
    return Pipeline([("preprocess", build_preprocessor(roles)), ("model", model_cls(**params))])


class LabelEncodedClassifier:
    """Meta-estimator that label-encodes y for estimators demanding integer
    classes (XGBoost >= 1.6 refuses string labels). Module-level so joblib
    pipelines deserialize anywhere mlteam is installed."""

    _estimator_type = "classifier"  # sklearn <= 1.5 protocol

    def __init__(self, estimator: Any = None) -> None:
        self.estimator = estimator

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {"estimator": self.estimator}
        if deep and hasattr(self.estimator, "get_params"):
            for key, value in self.estimator.get_params(deep=True).items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **params: Any) -> LabelEncodedClassifier:
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        nested = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("estimator__")}
        if nested:
            self.estimator.set_params(**nested)
        return self

    def fit(self, X: Any, y: Any) -> LabelEncodedClassifier:
        from sklearn.base import clone
        from sklearn.preprocessing import LabelEncoder

        self._encoder = LabelEncoder().fit(y)
        self.classes_ = self._encoder.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, self._encoder.transform(y))
        return self

    def predict(self, X: Any) -> Any:
        return self._encoder.inverse_transform(self.estimator_.predict(X))

    def predict_proba(self, X: Any) -> Any:
        return self.estimator_.predict_proba(X)

    def __sklearn_tags__(self) -> Any:  # sklearn >= 1.6 protocol
        from sklearn.base import BaseEstimator, ClassifierMixin

        class _Default(ClassifierMixin, BaseEstimator):
            pass

        return _Default().__sklearn_tags__()


class CatBoostCategoricalWrapper:
    """Passes ``cat_features`` at fit() time instead of the CatBoost
    constructor, so ``sklearn.base.clone`` keeps working.

    CatBoost's estimators implement their own get_params/set_params (they do
    not subclass sklearn's BaseEstimator) and, at least on the sklearn
    version this repo pins, sklearn's stricter clone() round-trip check
    raises "constructor either does not set or modifies parameter
    cat_features" whenever cat_features is passed to the CatBoost
    constructor at all -- reproduced directly against catboost+sklearn in
    this environment, independent of *how* cat_features is supplied
    (constructor kwarg or post-construction set_params both fail identically).
    Every CV/HPO path in this codebase clones the estimator, so a template
    that cannot survive clone() is not shippable. Keeping cat_features off
    the wrapped estimator's constructor entirely and forwarding it at fit()
    (which CatBoost's .fit() accepts directly) sidesteps the incompatibility
    without touching CatBoost's own get_params/set_params, since this
    wrapper's own params are the only ones sklearn's clone() ever inspects.
    """

    def __init__(self, estimator: Any = None, cat_features: tuple[int, ...] = ()) -> None:
        self.estimator = estimator
        self.cat_features = cat_features

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {"estimator": self.estimator, "cat_features": self.cat_features}
        if deep and hasattr(self.estimator, "get_params"):
            for key, value in self.estimator.get_params(deep=True).items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **params: Any) -> CatBoostCategoricalWrapper:
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        if "cat_features" in params:
            self.cat_features = params.pop("cat_features")
        nested = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("estimator__")}
        if nested:
            self.estimator.set_params(**nested)
        return self

    def fit(self, X: Any, y: Any) -> CatBoostCategoricalWrapper:
        from sklearn.base import clone

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, cat_features=list(self.cat_features))
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: Any) -> Any:
        import numpy as np

        return np.asarray(self.estimator_.predict(X)).ravel()

    def predict_proba(self, X: Any) -> Any:
        return self.estimator_.predict_proba(X)

    def __sklearn_tags__(self) -> Any:  # sklearn >= 1.6 protocol
        from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

        is_classifier = getattr(self.estimator, "_estimator_type", None) == "classifier"
        mixin = ClassifierMixin if is_classifier else RegressorMixin

        class _Default(mixin, BaseEstimator):
            pass

        return _Default().__sklearn_tags__()


def _build_xgboost(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier, XGBRegressor

    params: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": 0,
        **dict(config),
    }
    if task_type in _CLASSIFICATION:
        model: Any = LabelEncodedClassifier(XGBClassifier(**params))
    else:
        model = XGBRegressor(**params)
    return Pipeline([("preprocess", build_preprocessor(roles)), ("model", model)])


def _build_catboost(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from catboost import CatBoostClassifier, CatBoostRegressor
    from sklearn.pipeline import Pipeline

    # cat_features by POSITION: build_catboost_preprocessor's ColumnTransformer
    # always emits the numeric block first, then the categorical block, in
    # roles order (audit L2) -- these are the columns CatBoost should treat
    # as categorical rather than ordinal/one-hot encoding away.
    cat_feature_indices = tuple(
        range(len(roles.numeric), len(roles.numeric) + len(roles.categorical))
    )
    params: dict[str, Any] = {
        "iterations": 300,
        "random_seed": seed,
        "verbose": 0,
        "allow_writing_files": False,
        "thread_count": 1,
        **dict(config),
    }
    model_cls = CatBoostClassifier if task_type in _CLASSIFICATION else CatBoostRegressor
    # cat_features is NOT passed to the constructor -- see
    # CatBoostCategoricalWrapper's docstring for why that breaks
    # sklearn.clone(). The wrapper forwards it at fit() time instead.
    model = CatBoostCategoricalWrapper(model_cls(**params), cat_features=cat_feature_indices)
    return Pipeline([("preprocess", build_catboost_preprocessor(roles)), ("model", model)])


def _build_random_forest(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.pipeline import Pipeline

    params: dict[str, Any] = {
        "n_estimators": 300,
        "random_state": seed,
        "n_jobs": 1,
        **dict(config),
    }
    model_cls = RandomForestClassifier if task_type in _CLASSIFICATION else RandomForestRegressor
    return Pipeline([("preprocess", build_preprocessor(roles)), ("model", model_cls(**params))])


def _build_extra_trees(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.pipeline import Pipeline

    params: dict[str, Any] = {
        "n_estimators": 300,
        "random_state": seed,
        "n_jobs": 1,
        **dict(config),
    }
    model_cls = ExtraTreesClassifier if task_type in _CLASSIFICATION else ExtraTreesRegressor
    return Pipeline([("preprocess", build_preprocessor(roles)), ("model", model_cls(**params))])


def _build_adaboost(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
    from sklearn.pipeline import Pipeline

    params: dict[str, Any] = {"n_estimators": 200, "random_state": seed, **dict(config)}
    model_cls = AdaBoostClassifier if task_type in _CLASSIFICATION else AdaBoostRegressor
    return Pipeline([("preprocess", build_preprocessor(roles)), ("model", model_cls(**params))])


def _build_linear(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline

    if task_type in _CLASSIFICATION:
        params: dict[str, Any] = {"max_iter": 2000, "random_state": seed, **dict(config)}
        model: Any = LogisticRegression(**params)
    else:
        params = {"alpha": 1.0, "random_state": seed, **dict(config)}
        model = Ridge(**params)
    return Pipeline([("preprocess", build_scaled_preprocessor(roles)), ("model", model)])


def _build_svm(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC, SVR

    if task_type in _CLASSIFICATION:
        # probability=True enables predict_proba for roc_auc/log_loss scorers.
        params: dict[str, Any] = {"probability": True, "random_state": seed, **dict(config)}
        model: Any = SVC(**params)
    else:
        model = SVR(**dict(config))
    return Pipeline([("preprocess", build_scaled_preprocessor(roles)), ("model", model)])


def _build_knn(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import Pipeline

    params: dict[str, Any] = {"n_neighbors": 15, "n_jobs": 1, **dict(config)}
    model_cls = KNeighborsClassifier if task_type in _CLASSIFICATION else KNeighborsRegressor
    model = model_cls(**params)
    return Pipeline([("preprocess", build_scaled_preprocessor(roles)), ("model", model)])


def _build_naive_bayes(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline

    model = GaussianNB(**dict(config))
    return Pipeline([("preprocess", build_scaled_preprocessor(roles)), ("model", model)])


def _build_mlp(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline

    params: dict[str, Any] = {
        "hidden_layer_sizes": (64, 32),
        "max_iter": 500,
        "early_stopping": True,
        "random_state": seed,
        **dict(config),
    }
    model_cls = MLPClassifier if task_type in _CLASSIFICATION else MLPRegressor
    model = model_cls(**params)
    return Pipeline([("preprocess", build_scaled_preprocessor(roles)), ("model", model)])


class UnsupervisedAnomalyClassifier:
    """Adapts an unsupervised outlier detector to the supervised interface.

    Fits the detector on NORMAL rows only (semi-supervised novelty setup —
    train labels exist on this platform, so use them), then scores all rows:
    ``predict_proba`` exposes min-max-scaled anomaly scores so ranking metrics
    (pr_auc/roc_auc) evaluate honestly against the held-back labels, and
    ``predict`` thresholds at the training contamination rate.
    """

    _estimator_type = "classifier"  # sklearn <= 1.5 protocol

    def __init__(self, detector: Any = None) -> None:
        self.detector = detector

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {"detector": self.detector}
        if deep and hasattr(self.detector, "get_params"):
            for key, value in self.detector.get_params(deep=True).items():
                params[f"detector__{key}"] = value
        return params

    def set_params(self, **params: Any) -> UnsupervisedAnomalyClassifier:
        if "detector" in params:
            self.detector = params.pop("detector")
        nested = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("detector__")}
        if nested:
            self.detector.set_params(**nested)
        return self

    def fit(self, X: Any, y: Any) -> UnsupervisedAnomalyClassifier:
        import numpy as np
        from sklearn.base import clone

        y_arr = np.asarray(y)
        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) != 2:
            raise ValueError(
                f"anomaly detection requires exactly 2 label values, got {len(classes)}"
            )
        self.classes_ = classes
        # The RARE label is the anomaly by definition.
        self._anomaly_label = classes[int(np.argmin(counts))]
        contamination = counts.min() / counts.sum()
        normal_mask = y_arr != self._anomaly_label
        X_arr = np.asarray(X)
        self.detector_ = clone(self.detector)
        self.detector_.fit(X_arr[normal_mask])
        train_scores = self._raw_scores(X_arr)
        self._score_min = float(train_scores.min())
        self._score_max = float(train_scores.max())
        self._threshold = float(np.quantile(train_scores, 1.0 - contamination))
        return self

    def _raw_scores(self, X: Any) -> Any:
        # decision_function: higher = more normal, for every sklearn detector.
        return -self.detector_.decision_function(X)

    def predict_proba(self, X: Any) -> Any:
        import numpy as np

        scores = self._raw_scores(np.asarray(X))
        span = max(self._score_max - self._score_min, 1e-12)
        p_anomaly = np.clip((scores - self._score_min) / span, 0.0, 1.0)
        proba = np.empty((len(p_anomaly), 2))
        anomaly_col = int(np.argmax(self.classes_ == self._anomaly_label))
        proba[:, anomaly_col] = p_anomaly
        proba[:, 1 - anomaly_col] = 1.0 - p_anomaly
        return proba

    def predict(self, X: Any) -> Any:
        import numpy as np

        scores = self._raw_scores(np.asarray(X))
        normal_label = self.classes_[self.classes_ != self._anomaly_label][0]
        return np.where(scores > self._threshold, self._anomaly_label, normal_label)

    def __sklearn_tags__(self) -> Any:  # sklearn >= 1.6 protocol
        from sklearn.base import BaseEstimator, ClassifierMixin

        class _Default(ClassifierMixin, BaseEstimator):
            pass

        return _Default().__sklearn_tags__()


def _build_isolation_forest(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline

    params: dict[str, Any] = {
        "n_estimators": 200,
        "random_state": seed,
        "n_jobs": 1,
        **dict(config),
    }
    model = UnsupervisedAnomalyClassifier(IsolationForest(**params))
    return Pipeline([("preprocess", build_preprocessor(roles)), ("model", model)])


def _build_one_class_svm(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.pipeline import Pipeline
    from sklearn.svm import OneClassSVM

    params: dict[str, Any] = {"nu": 0.1, **dict(config)}
    model = UnsupervisedAnomalyClassifier(OneClassSVM(**params))
    return Pipeline([("preprocess", build_scaled_preprocessor(roles)), ("model", model)])


def _build_local_outlier_factor(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.pipeline import Pipeline

    # novelty=True is required to score unseen rows (CV/holdout/serving).
    params: dict[str, Any] = {"n_neighbors": 20, "novelty": True, **dict(config)}
    model = UnsupervisedAnomalyClassifier(LocalOutlierFactor(**params))
    return Pipeline([("preprocess", build_scaled_preprocessor(roles)), ("model", model)])


def _build_stacking(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    """Stacked ensemble of the strongest families. Deliberately config-free:
    the vetted base models are the tuning surface, not the stack itself."""
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
        StackingClassifier,
        StackingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier, XGBRegressor

    lgbm_params: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "random_state": seed,
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": 1,
        "verbosity": -1,
    }
    xgb_params: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": 0,
    }
    rf_params: dict[str, Any] = {"n_estimators": 200, "random_state": seed, "n_jobs": 1}

    if task_type in _CLASSIFICATION:
        model: Any = StackingClassifier(
            estimators=[
                ("lgbm", LGBMClassifier(**lgbm_params)),
                ("xgb", LabelEncodedClassifier(XGBClassifier(**xgb_params))),
                ("rf", RandomForestClassifier(**rf_params)),
            ],
            final_estimator=LogisticRegression(max_iter=2000, random_state=seed),
            cv=3,
            n_jobs=1,
        )
    else:
        model = StackingRegressor(
            estimators=[
                ("lgbm", LGBMRegressor(**lgbm_params)),
                ("xgb", XGBRegressor(**xgb_params)),
                ("rf", RandomForestRegressor(**rf_params)),
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=1,
        )
    return Pipeline([("preprocess", build_preprocessor(roles)), ("model", model)])


def _build_stacking_deep(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    """AutoGluon-style depth: 5 diverse base families, OOF meta-features via
    internal CV, plus passthrough so the meta-learner sees the raw features
    too. Config-free by design — the vetted bases are the tuning surface."""
    from catboost import CatBoostClassifier, CatBoostRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        StackingClassifier,
        StackingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier, XGBRegressor

    lgbm: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "random_state": seed,
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": 1,
        "verbosity": -1,
    }
    xgb: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": 0,
    }
    cat: dict[str, Any] = {
        "iterations": 200,
        "random_seed": seed,
        "verbose": 0,
        "allow_writing_files": False,
        "thread_count": 1,
    }
    forest: dict[str, Any] = {"n_estimators": 200, "random_state": seed, "n_jobs": 1}

    if task_type in _CLASSIFICATION:
        model: Any = StackingClassifier(
            estimators=[
                ("lgbm", LGBMClassifier(**lgbm)),
                ("xgb", LabelEncodedClassifier(XGBClassifier(**xgb))),
                ("cat", CatBoostClassifier(**cat)),
                ("rf", RandomForestClassifier(**forest)),
                ("et", ExtraTreesClassifier(**forest)),
            ],
            final_estimator=LogisticRegression(max_iter=2000, random_state=seed),
            cv=3,
            passthrough=True,
            n_jobs=1,
        )
    else:
        model = StackingRegressor(
            estimators=[
                ("lgbm", LGBMRegressor(**lgbm)),
                ("xgb", XGBRegressor(**xgb)),
                ("cat", CatBoostRegressor(**cat)),
                ("rf", RandomForestRegressor(**forest)),
                ("et", ExtraTreesRegressor(**forest)),
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            passthrough=True,
            n_jobs=1,
        )
    return Pipeline([("preprocess", build_preprocessor(roles)), ("model", model)])


class SeasonalNaiveRegressor:
    """The honest forecasting floor: predict the value one season ago.

    Reads the ``*_lag_{season_length}`` column the time-feature engineer
    produced — no learning at all. A forecaster that cannot beat this is
    worse than a lookup table, whatever its CV score against dummy-mean says.
    """

    _estimator_type = "regressor"  # sklearn <= 1.5 protocol

    def __init__(self, season_length: int = 7) -> None:
        self.season_length = season_length

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"season_length": self.season_length}

    def set_params(self, **params: Any) -> SeasonalNaiveRegressor:
        if "season_length" in params:
            self.season_length = params["season_length"]
        return self

    def fit(self, X: Any, y: Any) -> SeasonalNaiveRegressor:
        suffix = f"_lag_{self.season_length}"
        matches = [c for c in getattr(X, "columns", []) if str(c).endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(
                f"seasonal naive needs exactly one '*{suffix}' lag column from the "
                f"time-feature engineer, found {matches or 'none'} — is this a "
                "forecasting run with lag features?"
            )
        self.lag_column_ = matches[0]
        import numpy as np

        self.fallback_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict(self, X: Any) -> Any:
        import numpy as np

        values = np.asarray(X[self.lag_column_], dtype=float)
        return np.where(np.isfinite(values), values, self.fallback_)

    def __sklearn_tags__(self) -> Any:  # sklearn >= 1.6 protocol
        from sklearn.base import BaseEstimator, RegressorMixin

        class _Default(RegressorMixin, BaseEstimator):
            pass

        return _Default().__sklearn_tags__()


def _build_seasonal_naive(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    # No preprocessor: the estimator reads one named lag column from the raw
    # frame; a ColumnTransformer would strip the names it needs.
    params: dict[str, Any] = {"season_length": 7, **dict(config)}
    return SeasonalNaiveRegressor(**params)


class QuantileGBMRegressor:
    """Three quantile-objective LightGBMs (default 5%/50%/95%).

    ``predict`` returns the median so every ordinary metric works unchanged;
    ``predict_quantiles`` exposes the full band for interval-aware reports
    and serving ("sales will be 90-110" beats "sales will be 100").
    """

    _estimator_type = "regressor"  # sklearn <= 1.5 protocol

    def __init__(
        self, quantiles: tuple[float, ...] = (0.05, 0.5, 0.95), **lgbm_params: Any
    ) -> None:
        self.quantiles = quantiles
        self.lgbm_params = lgbm_params

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"quantiles": self.quantiles, **self.lgbm_params}

    def set_params(self, **params: Any) -> QuantileGBMRegressor:
        if "quantiles" in params:
            self.quantiles = params.pop("quantiles")
        self.lgbm_params.update(params)
        return self

    def fit(self, X: Any, y: Any) -> QuantileGBMRegressor:
        from lightgbm import LGBMRegressor

        if 0.5 not in self.quantiles:
            raise ValueError("quantiles must include 0.5 — predict() returns the median")
        self.models_ = {
            q: LGBMRegressor(objective="quantile", alpha=q, **self.lgbm_params).fit(X, y)
            for q in self.quantiles
        }
        return self

    def predict(self, X: Any) -> Any:
        return self.models_[0.5].predict(X)

    def predict_quantiles(self, X: Any) -> Any:
        """Columns ordered by ascending quantile; sorted per-row at predict.

        Each quantile is an independently trained LightGBM, so nothing
        constrains their outputs to be monotone in q — the 5th-percentile
        model can (and on real data sometimes does) predict above the
        50th-percentile model for a given row. Reporting a "90% interval"
        whose lower bound exceeds its upper bound is a worse user-facing bug
        than the crossing itself, so rows are sorted ascending across the
        quantile axis, which is a standard practical fix (interval-only,
        does not change the median predict() returns).
        """
        import numpy as np

        stacked = np.column_stack([self.models_[q].predict(X) for q in sorted(self.quantiles)])
        return np.sort(stacked, axis=1)

    def __sklearn_tags__(self) -> Any:  # sklearn >= 1.6 protocol
        from sklearn.base import BaseEstimator, RegressorMixin

        class _Default(RegressorMixin, BaseEstimator):
            pass

        return _Default().__sklearn_tags__()


def _build_lightgbm_quantile(
    task_type: TaskType, roles: ColumnRoles, config: Mapping[str, Any], seed: int
) -> Any:
    from sklearn.pipeline import Pipeline

    params: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "random_state": seed,
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": 1,
        "verbosity": -1,
        **dict(config),
    }
    model = QuantileGBMRegressor(**params)
    return Pipeline([("preprocess", build_preprocessor(roles)), ("model", model)])


_TEMPLATES: dict[str, PipelineTemplate] = {
    template.template_id: template
    for template in (
        PipelineTemplate(
            template_id="dummy_baseline",
            description="Prior/mean dummy model — the floor every model must beat",
            task_types=_ALL_TASKS,
            is_baseline=True,
            build=_build_dummy,
        ),
        PipelineTemplate(
            template_id="lightgbm_default",
            description="LightGBM with sane defaults — the standard tabular baseline",
            task_types=_ALL_TASKS,
            is_baseline=True,
            build=_build_lightgbm,
        ),
        PipelineTemplate(
            template_id="tabpfn_default",
            description="TabPFN v2 — strong zero-tuning baseline for small/medium data",
            task_types=_ALL_TASKS,
            is_baseline=True,
            build=_build_tabpfn,
        ),
        PipelineTemplate(
            template_id="xgboost_default",
            description="XGBoost — strong gradient-boosted trees, robust general candidate",
            task_types=_ALL_TASKS,
            is_baseline=False,
            build=_build_xgboost,
        ),
        PipelineTemplate(
            template_id="catboost_default",
            description="CatBoost — gradient boosting that excels on categorical-heavy data",
            task_types=_ALL_TASKS,
            is_baseline=False,
            build=_build_catboost,
        ),
        PipelineTemplate(
            template_id="random_forest_default",
            description="Random forest — robust bagged trees, low overfitting risk",
            task_types=_ALL_TASKS,
            is_baseline=False,
            build=_build_random_forest,
        ),
        PipelineTemplate(
            template_id="extra_trees_default",
            description="Extremely randomized trees — faster RF variant, high variance data",
            task_types=_ALL_TASKS,
            is_baseline=False,
            build=_build_extra_trees,
        ),
        PipelineTemplate(
            template_id="adaboost_default",
            description="AdaBoost — classic boosting, decent on clean low-noise data",
            task_types=_ALL_TASKS,
            is_baseline=False,
            build=_build_adaboost,
        ),
        PipelineTemplate(
            template_id="linear_default",
            description="Regularized linear model (logistic/ridge) — interpretable, "
            "great when relationships are near-linear or n_features is high",
            task_types=_ALL_TASKS,
            is_baseline=False,
            build=_build_linear,
        ),
        PipelineTemplate(
            template_id="svm_default",
            description="Support vector machine (RBF) — small/medium datasets only "
            "(quadratic in rows); strong decision boundaries",
            task_types=_ALL_TASKS,
            is_baseline=False,
            build=_build_svm,
        ),
        PipelineTemplate(
            template_id="knn_default",
            description="K-nearest neighbors — small datasets, local structure",
            task_types=_ALL_TASKS,
            is_baseline=False,
            build=_build_knn,
        ),
        PipelineTemplate(
            template_id="naive_bayes_default",
            description="Gaussian naive Bayes — very fast probabilistic classifier",
            task_types=_CLASSIFICATION,
            is_baseline=False,
            build=_build_naive_bayes,
        ),
        PipelineTemplate(
            template_id="mlp_default",
            description="Multi-layer perceptron — neural net for larger numeric-heavy data",
            task_types=_ALL_TASKS,
            is_baseline=False,
            build=_build_mlp,
        ),
        PipelineTemplate(
            template_id="stacking_ensemble_default",
            description="Stacked LightGBM+XGBoost+RF with a linear meta-learner — "
            "the accuracy ceiling-pusher; slower, use when metrics matter most",
            task_types=_ALL_TASKS,
            is_baseline=False,
            build=_build_stacking,
        ),
        PipelineTemplate(
            template_id="isolation_forest_default",
            description="Isolation forest — unsupervised anomaly scorer fit on normal "
            "rows, evaluated against labels; robust default for anomaly detection",
            task_types=_ANOMALY_ONLY,
            is_baseline=False,
            build=_build_isolation_forest,
        ),
        PipelineTemplate(
            template_id="one_class_svm_default",
            description="One-class SVM — boundary-based novelty detector, "
            "small/medium data",
            task_types=_ANOMALY_ONLY,
            is_baseline=False,
            build=_build_one_class_svm,
        ),
        PipelineTemplate(
            template_id="local_outlier_factor_default",
            description="Local outlier factor — density-based novelty detector, "
            "catches local anomalies globals miss",
            task_types=_ANOMALY_ONLY,
            is_baseline=False,
            build=_build_local_outlier_factor,
        ),
        PipelineTemplate(
            template_id="stacking_deep_default",
            description="Deep stack: 5 base families (LGBM/XGB/CatBoost/RF/ExtraTrees) "
            "+ passthrough meta — the strongest (and slowest) accuracy pusher",
            task_types=_ALL_TASKS,
            is_baseline=False,
            build=_build_stacking_deep,
        ),
        PipelineTemplate(
            template_id="seasonal_naive_default",
            description="Seasonal naive — predicts the value one season ago; the "
            "honest forecasting floor every real forecaster must beat",
            task_types=frozenset({TaskType.FORECASTING}),
            is_baseline=False,
            build=_build_seasonal_naive,
        ),
        PipelineTemplate(
            template_id="lightgbm_quantile_default",
            description="LightGBM quantile trio (5/50/95%) — point forecast plus "
            "prediction intervals for risk-aware decisions",
            task_types=_REGRESSION_LIKE,
            is_baseline=False,
            build=_build_lightgbm_quantile,
        ),
    )
}


def get_template(template_id: str) -> PipelineTemplate:
    template = _TEMPLATES.get(template_id)
    if template is None:
        raise KeyError(f"unknown template '{template_id}'; registered: {sorted(_TEMPLATES)}")
    return template


# Whitelist, not free-form (JUDGMENT_ROADMAP.md Task 2): log1p over a bare
# log because it tolerates exact zeros (log(0) is -inf; log1p(0) is 0), which
# matters for price-like/count-like targets that legitimately hit zero.
# Requires targets >= 0 (enforced loudly, not silently clipped) since log1p
# of a negative value below -1 is undefined and between -1 and 0 silently
# produces a valid-looking but meaningless negative output.
TARGET_TRANSFORMS = frozenset({"log1p"})


def wrap_target_transform(pipeline: Any, transform: str | None, task_type: TaskType) -> Any:
    """Wrap ``pipeline`` in a ``TransformedTargetRegressor`` for ``transform``.

    Returns ``pipeline`` unchanged for ``None`` (the overwhelmingly common
    case — most targets need no transform). Wrapping the whole pipeline
    (preprocessing + model) rather than just the final estimator means
    every sklearn scorer, CV metric, holdout metric, and baseline-delta
    computation calls ``.predict()`` and automatically gets ORIGINAL-unit
    predictions back (``inverse_func`` runs inside ``predict()``) — no
    metric plumbing anywhere else needs to know a transform happened.

    Only ``fit``/``predict``/``score`` are proxied by
    ``TransformedTargetRegressor``; a template whose estimator exposes
    extra methods (e.g. ``QuantileGBMRegressor.predict_quantiles``) loses
    access to them once wrapped — out of scope here since target_transform
    is opt-in per experiment, never a template-wide default.
    """
    if transform is None:
        return pipeline
    if task_type is not TaskType.REGRESSION:
        raise ValueError(
            f"target_transform is only supported for {TaskType.REGRESSION.value} "
            f"(FORECASTING is excluded: lag/rolling features are computed from the "
            f"raw target pre-split, so mixing raw-scale features with transformed-"
            f"scale fitting is a correctness question deferred deliberately); "
            f"got task_type={task_type.value}"
        )
    if transform not in TARGET_TRANSFORMS:
        raise ValueError(
            f"unknown target_transform '{transform}'; supported: {sorted(TARGET_TRANSFORMS)}"
        )

    from sklearn.compose import TransformedTargetRegressor

    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def list_templates(task_type: TaskType | None = None) -> tuple[PipelineTemplate, ...]:
    templates = _TEMPLATES.values()
    if task_type is not None:
        templates = (t for t in templates if t.supports(task_type))
    return tuple(templates)
