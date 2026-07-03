"""Requirement Understanding agent: natural-language goal → ProblemSpec.

The LLM drafts the spec; deterministic post-validation pins it to reality
(target column must exist, metric must be computable for the task type).
The result awaits HUMAN GATE 1 — ``human_confirmed`` is always False here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mlteam.contracts import CostAsymmetry, ProblemSpec, TaskType, new_artifact
from mlteam.evaluation.metrics import scoring_for
from mlteam.llm.client import LLMClient
from mlteam.store.interface import ArtifactStore

SYSTEM_PROMPT = """You are the Requirement Understanding agent of an autonomous \
ML engineering team. Convert the user's business goal into a precise machine \
learning problem specification.

Rules:
- Choose the target column ONLY from the provided dataset columns.
- Pick task_type from the allowed values based on the goal and target semantics.
- primary_metric must fit the task: binary -> roc_auc/accuracy/f1_macro/log_loss; \
multiclass -> accuracy/f1_macro/log_loss; regression -> rmse/mae/r2.
- cost_asymmetry: express the business cost ratio of false positives vs false \
negatives if the goal implies one; otherwise leave both at 1.0.
- Record assumptions you had to make. Be conservative: do not invent constraints \
the user never stated.
Return only JSON matching the schema."""


class CostAsymmetryDraft(BaseModel):
    false_positive_cost: float = 1.0
    false_negative_cost: float = 1.0
    rationale: str = ""


class ProblemSpecDraft(BaseModel):
    """LLM output schema — payload of a ProblemSpec, nothing more."""

    task_type: TaskType
    target_column: str
    prediction_horizon: str | None = None
    primary_metric: str
    secondary_metrics: list[str] = Field(default_factory=list)
    business_constraints: list[str] = Field(default_factory=list)
    cost_asymmetry: CostAsymmetryDraft = CostAsymmetryDraft()
    assumptions: list[str] = Field(default_factory=list)


def understand_requirements(
    client: LLMClient,
    store: ArtifactStore,
    *,
    goal_text: str,
    dataset_columns: list[str],
    run_id: str,
) -> ProblemSpec:
    """Draft, validate, and store a ProblemSpec (unconfirmed until Gate 1)."""

    def _post_validate(draft: ProblemSpecDraft) -> None:
        if draft.target_column not in dataset_columns:
            raise ValueError(
                f"target_column '{draft.target_column}' is not in the dataset; "
                f"choose one of: {dataset_columns}"
            )
        valid_metrics = set(scoring_for(draft.task_type))
        if draft.primary_metric not in valid_metrics:
            raise ValueError(
                f"primary_metric '{draft.primary_metric}' is invalid for "
                f"{draft.task_type.value}; choose one of: {sorted(valid_metrics)}"
            )

    result = client.structured(
        agent="requirement-understanding",
        system_prompt=SYSTEM_PROMPT,
        user_payload={
            "business_goal": goal_text,
            "dataset_columns": dataset_columns,
        },
        schema=ProblemSpecDraft,
        post_validate=_post_validate,
    )
    draft = result.output
    spec = new_artifact(
        ProblemSpec,
        producer_agent="requirement-understanding",
        run_id=run_id,
        task_type=draft.task_type,
        target_column=draft.target_column,
        prediction_horizon=draft.prediction_horizon,
        primary_metric=draft.primary_metric,
        secondary_metrics=tuple(draft.secondary_metrics),
        business_constraints=tuple(draft.business_constraints),
        cost_asymmetry=CostAsymmetry(
            false_positive_cost=draft.cost_asymmetry.false_positive_cost,
            false_negative_cost=draft.cost_asymmetry.false_negative_cost,
            rationale=draft.cost_asymmetry.rationale,
        ),
        assumptions=tuple(draft.assumptions),
        human_confirmed=False,
    )
    store.put(spec)
    return spec
