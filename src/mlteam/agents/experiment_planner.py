"""Experiment Planner agent: which templates to train, with budget estimates.

The LLM selects templates + configs from the vetted registry; deterministic
code computes budget estimates and enforces the structural rules (cheap
baselines ALWAYS first — the ExperimentPlan validator refuses anything else).
The plan then awaits HUMAN GATE 2 with its cost estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from mlteam.contracts import (
    BudgetEstimate,
    DataProfile,
    ExperimentPlan,
    ExperimentSpec,
    ProblemSpec,
    new_artifact,
)
from mlteam.llm.client import LLMClient
from mlteam.llm.summaries import profile_summary
from mlteam.store.interface import ArtifactStore
from mlteam.templates.registry import get_template, list_templates

SYSTEM_PROMPT = """You are the Experiment Planner of an autonomous ML engineering \
team. Select which vetted pipeline templates to train and with what config.

Rules:
- template_id must come from the provided registry list. You never write model \
code — you only pick templates and fill configs.
- The plan MUST start with the cheap baselines, in this order: dummy_baseline \
first, then lightgbm_default — both marked is_baseline=true. They are the sanity \
floor every candidate must beat.
- After the baselines, add 2-4 candidate experiments chosen for THIS dataset. \
Guidance: xgboost_default and catboost_default are strong general candidates \
(catboost especially on categorical-heavy data); tabpfn_default excels under \
~10k rows; random_forest_default/extra_trees_default are robust to noise; \
linear_default when relationships look near-linear or interpretability matters; \
svm_default and knn_default only on small datasets (<5k rows); mlp_default on \
larger numeric-heavy data; naive_bayes_default is classification-only; \
stacking_ensemble_default pushes the accuracy ceiling when budget allows; for \
anomaly_detection also consider the dedicated detectors (isolation_forest_default, \
one_class_svm_default, local_outlier_factor_default) alongside the classifiers.
- Prefer an EMPTY config — every template ships vetted defaults, and HPO tunes \
the winner later. Only set config keys when the profile clearly demands it, and \
only keys valid for that model's constructor.
- rationale should reference the profile (size, imbalance, cardinality).
Return only JSON matching the schema."""

# Rough wall-time heuristics (seconds) per 1k rows, by template.
_SECONDS_PER_1K_ROWS: dict[str, float] = {
    "dummy_baseline": 0.5,
    "lightgbm_default": 6.0,
    "tabpfn_default": 45.0,
    "xgboost_default": 8.0,
    "catboost_default": 15.0,
    "random_forest_default": 10.0,
    "extra_trees_default": 8.0,
    "adaboost_default": 8.0,
    "linear_default": 2.0,
    "svm_default": 30.0,  # quadratic in rows — heuristic assumes small data
    "knn_default": 3.0,
    "naive_bayes_default": 1.0,
    "mlp_default": 20.0,
    "stacking_ensemble_default": 45.0,  # 3 base models x internal cv=3
    "isolation_forest_default": 4.0,
    "one_class_svm_default": 20.0,
    "local_outlier_factor_default": 5.0,
}
_MIN_BUDGET_SECONDS = 2.0


class ModelConfigDraft(BaseModel):
    """Whitelisted hyperparameters — the LLM tunes within these knobs only.

    (The structured-output grammar cannot express open dicts; a typed subset
    is also exactly what 'agents decide, libraries execute' wants.) Knobs span
    the template families: boosting (n_estimators/learning_rate/...), CatBoost
    (iterations/depth/l2_leaf_reg), forests (min_samples_leaf/max_features),
    linear/SVM (C/alpha), KNN (n_neighbors). Unused knobs stay None.
    """

    n_estimators: int | None = None
    learning_rate: float | None = None
    max_depth: int | None = None
    num_leaves: int | None = None
    min_child_samples: int | None = None
    subsample: float | None = None
    colsample_bytree: float | None = None
    reg_alpha: float | None = None
    reg_lambda: float | None = None
    iterations: int | None = None
    depth: int | None = None
    l2_leaf_reg: float | None = None
    min_samples_leaf: int | None = None
    max_features: float | None = None
    C: float | None = None
    alpha: float | None = None
    n_neighbors: int | None = None

    def to_config(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ExperimentDraft(BaseModel):
    experiment_id: str
    template_id: str
    config: ModelConfigDraft = Field(default_factory=ModelConfigDraft)
    is_baseline: bool = False


class ExperimentPlanDraft(BaseModel):
    experiments: list[ExperimentDraft] = Field(min_length=1)
    rationale: str


def estimate_budget(template_id: str, n_rows: int) -> BudgetEstimate:
    """Deterministic wall-time estimate; the LLM never invents budgets."""
    per_1k = _SECONDS_PER_1K_ROWS.get(template_id, 10.0)
    seconds = max(_MIN_BUDGET_SECONDS, per_1k * (n_rows / 1000.0))
    return BudgetEstimate(wall_time_seconds=round(seconds, 1))


REVISION_PROMPT = """You are the Experiment Planner of an autonomous ML \
engineering team, revising the plan after a Critic verdict of \
iterate_with_actions.

Rules:
- Propose 1-3 NEW candidate experiments that concretely address the Critic's \
actions. Do not repeat an experiment_id or an identical config that was \
already trained.
- template_id must come from the registry list; is_baseline must be false \
(baselines already exist).
- config keys must be valid model constructor parameters. Ground every choice \
in the critique actions and prior results.
- rationale must say which critique action each experiment addresses.
Return only JSON matching the schema."""


class RevisionDraft(BaseModel):
    experiments: list[ExperimentDraft] = Field(min_length=1, max_length=3)
    rationale: str


@dataclass(frozen=True)
class RevisionResult:
    """Revised candidates for one iteration. NOT an ExperimentPlan artifact —
    that contract mandates baselines-first for full plans; a revision only
    adds candidates, and its provenance flows through the trained models."""

    specs: tuple[ExperimentSpec, ...]
    rationale: str


def revise_experiment_plan(
    client: LLMClient,
    *,
    problem: ProblemSpec,
    critique_actions: tuple[str, ...],
    prior_results: list[dict[str, Any]],
    prior_experiment_ids: set[str],
    n_rows: int,
) -> RevisionResult:
    """Ask the planner for 1-3 new candidates addressing the critique."""
    registry = [
        {"template_id": t.template_id, "description": t.description}
        for t in list_templates(problem.task_type)
    ]
    registry_ids = {entry["template_id"] for entry in registry}

    def _post_validate(draft: RevisionDraft) -> None:
        for exp in draft.experiments:
            if exp.template_id not in registry_ids:
                raise ValueError(
                    f"template '{exp.template_id}' is not in the registry; "
                    f"choose from: {sorted(registry_ids)}"
                )
            if exp.experiment_id in prior_experiment_ids:
                raise ValueError(
                    f"experiment_id '{exp.experiment_id}' was already used; "
                    "pick a new id for the revised experiment"
                )
            if exp.is_baseline:
                raise ValueError("revision experiments must have is_baseline=false")

    result = client.structured(
        agent="experiment-planner-revision",
        system_prompt=REVISION_PROMPT,
        user_payload={
            "critique_actions": list(critique_actions),
            "prior_results": prior_results,
            "template_registry": registry,
            "used_experiment_ids": sorted(prior_experiment_ids),
        },
        schema=RevisionDraft,
        post_validate=_post_validate,
    )
    return RevisionResult(
        specs=tuple(
            ExperimentSpec(
                experiment_id=e.experiment_id,
                template_id=e.template_id,
                config=e.config.to_config(),
                budget=estimate_budget(e.template_id, n_rows),
                is_baseline=False,
            )
            for e in result.output.experiments
        ),
        rationale=result.output.rationale,
    )


def plan_experiments(
    client: LLMClient,
    store: ArtifactStore,
    *,
    profile: DataProfile,
    problem: ProblemSpec,
    n_rows: int,
    run_id: str,
) -> ExperimentPlan:
    """Draft, validate, and store an ExperimentPlan (unapproved until Gate 2)."""
    registry = [
        {"template_id": t.template_id, "description": t.description}
        for t in list_templates(problem.task_type)
    ]
    registry_ids = {entry["template_id"] for entry in registry}

    def _post_validate(draft: ExperimentPlanDraft) -> None:
        ids = [e.experiment_id for e in draft.experiments]
        if len(ids) != len(set(ids)):
            raise ValueError("experiment_id values must be unique")
        for exp in draft.experiments:
            if exp.template_id not in registry_ids:
                raise ValueError(
                    f"template '{exp.template_id}' is not in the registry; "
                    f"choose from: {sorted(registry_ids)}"
                )
            if not get_template(exp.template_id).supports(problem.task_type):
                raise ValueError(
                    f"template '{exp.template_id}' does not support {problem.task_type.value}"
                )
        first_two = [e.template_id for e in draft.experiments[:2]]
        if first_two[:1] != ["dummy_baseline"] or (
            len(first_two) < 2 or first_two[1] != "lightgbm_default"
        ):
            raise ValueError(
                "the plan must start with dummy_baseline then lightgbm_default, "
                "both with is_baseline=true"
            )
        if not all(e.is_baseline for e in draft.experiments[:2]):
            raise ValueError("the first two experiments must have is_baseline=true")

    result = client.structured(
        agent="experiment-planner",
        system_prompt=SYSTEM_PROMPT,
        user_payload={
            "problem": {
                "task_type": problem.task_type.value,
                "primary_metric": problem.primary_metric,
            },
            "n_rows": n_rows,
            "template_registry": registry,
            "profile": profile_summary(profile),
        },
        schema=ExperimentPlanDraft,
        post_validate=_post_validate,
    )
    draft = result.output
    plan = new_artifact(
        ExperimentPlan,
        producer_agent="experiment-planner",
        run_id=run_id,
        parent_ids=(profile.meta.artifact_id, problem.meta.artifact_id),
        experiments=tuple(
            ExperimentSpec(
                experiment_id=e.experiment_id,
                template_id=e.template_id,
                config=e.config.to_config(),
                budget=estimate_budget(e.template_id, n_rows),
                is_baseline=e.is_baseline,
            )
            for e in draft.experiments
        ),
        rationale=draft.rationale,
        human_approved=False,
    )
    store.put(plan)
    return plan
