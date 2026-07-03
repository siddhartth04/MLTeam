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

from mlteam.contracts import TaskType

_CLASSIFICATION = frozenset({TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION})
_ALL_TASKS = _CLASSIFICATION | {TaskType.REGRESSION}

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
    """Impute + ordinal-encode inside the pipeline (leakage-safe by design)."""
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
    )
}


def get_template(template_id: str) -> PipelineTemplate:
    template = _TEMPLATES.get(template_id)
    if template is None:
        raise KeyError(f"unknown template '{template_id}'; registered: {sorted(_TEMPLATES)}")
    return template


def list_templates(task_type: TaskType | None = None) -> tuple[PipelineTemplate, ...]:
    templates = _TEMPLATES.values()
    if task_type is not None:
        templates = (t for t in templates if t.supports(task_type))
    return tuple(templates)
