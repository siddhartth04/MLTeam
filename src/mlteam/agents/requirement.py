"""Requirement Understanding agent: natural-language goal → ProblemSpec.

The LLM drafts the spec; deterministic post-validation pins it to reality
(target column must exist, metric must be computable for the task type).
The result awaits HUMAN GATE 1 — ``human_confirmed`` is always False here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mlteam.agents.preview import ColumnPreview
from mlteam.contracts import CostAsymmetry, DecisionRecord, ProblemSpec, TaskType, new_artifact
from mlteam.evaluation.metrics import scoring_for
from mlteam.llm.client import LLMClient
from mlteam.store.interface import ArtifactStore

SYSTEM_PROMPT = """You are the Requirement Understanding agent of an autonomous \
ML engineering team. Convert the user's business goal into a precise machine \
learning problem specification.

Rules:
- Choose the target column ONLY from the provided dataset columns.
- Pick task_type from the allowed values based on the goal and target semantics. \
Use forecasting ONLY when the goal is to predict future values of a numeric \
series over time (e.g. "forecast next month's sales") — then also set \
time_column to the dataset's date/time column and prediction_horizon to the \
requested horizon.
- Use anomaly_detection when the goal is to catch rare labeled events: fraud, \
defects, intrusions, equipment failures. Predictive-maintenance goals map as: \
"will it fail within X" -> anomaly_detection (or binary_classification if \
failures are not rare); "remaining useful life / when will it fail" -> \
regression or forecasting.
- primary_metric must fit the task: binary -> roc_auc/pr_auc/accuracy/f1_macro/\
log_loss; multiclass -> accuracy/f1_macro/log_loss; regression or forecasting \
-> rmse/mae/r2; anomaly_detection -> pr_auc/roc_auc/f1_macro (prefer pr_auc — \
it is the honest metric for rare events).
- cost_asymmetry: express the business cost ratio of false positives vs false \
negatives if the goal implies one; otherwise leave both at 1.0.
- Record assumptions you had to make. Be conservative: do not invent constraints \
the user never stated.
- why_target: explain WHY this column is the target, not just that it is — e.g. \
its name/position matches the goal, it looks like the only outcome-shaped column.
- target_alternatives: list other column names that were plausible target \
candidates and were considered, even if quickly rejected. Empty only when \
truly no other column could plausibly be the target.
- target_evidence: list the CONCRETE signals that pointed to this target. When \
column_profile is provided, ground this in its stats — inferred_type matching \
the task (numeric for regression, low-cardinality text for classification), \
null_fraction (a target with many nulls is suspicious), n_unique (a near-1:1 \
n_unique to row count suggests an identifier, not a target), and sample_values. \
Also flag in target_evidence (and target_alternatives, if relevant) any OTHER \
column whose name or values look like a near-duplicate or derived version of \
the target — that is a leakage risk worth surfacing even at this early stage.
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
    time_column: str | None = None
    primary_metric: str
    secondary_metrics: list[str] = Field(default_factory=list)
    business_constraints: list[str] = Field(default_factory=list)
    cost_asymmetry: CostAsymmetryDraft = CostAsymmetryDraft()
    assumptions: list[str] = Field(default_factory=list)
    # WHY this target over alternatives — optional-with-defaults so replaying
    # fixtures recorded before these fields existed still validates.
    why_target: str = ""
    target_alternatives: list[str] = Field(default_factory=list)
    target_evidence: list[str] = Field(default_factory=list)


def understand_requirements(
    client: LLMClient,
    store: ArtifactStore,
    *,
    goal_text: str,
    dataset_columns: list[str],
    run_id: str,
    column_profile: list[ColumnPreview] | None = None,
) -> ProblemSpec:
    """Draft, validate, and store a ProblemSpec (unconfirmed until Gate 1).

    ``column_profile`` (from ``agents.preview.column_preview``) is optional so
    direct/test callers can still pass bare column names; when given, it lets
    the LLM rank target candidates on null rate, cardinality, and inferred
    type instead of choosing from names alone. ``dataset_columns`` remains
    the sole source of truth for the target-exists validation below,
    independent of whatever the profiler decided to compute.
    """

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
        if draft.time_column is not None and draft.time_column not in dataset_columns:
            raise ValueError(
                f"time_column '{draft.time_column}' is not in the dataset; "
                f"choose one of: {dataset_columns}"
            )

    payload: dict[str, Any] = {
        "business_goal": goal_text,
        "dataset_columns": dataset_columns,
    }
    if column_profile:
        payload["column_profile"] = [
            {
                "name": p.name,
                "inferred_type": p.inferred_type,
                "null_fraction": p.null_fraction,
                "n_unique": p.n_unique,
                "sample_values": list(p.sample_values),
            }
            for p in column_profile
        ]

    result = client.structured(
        agent="requirement-understanding",
        system_prompt=SYSTEM_PROMPT,
        user_payload=payload,
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
        time_column=draft.time_column,
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
        reasoning=DecisionRecord(
            decision=f"target_column = '{draft.target_column}' ({draft.task_type.value})",
            why=draft.why_target,
            evidence=tuple(draft.target_evidence),
            alternatives_considered=tuple(draft.target_alternatives),
            assumptions=tuple(draft.assumptions),
        ),
    )
    store.put(spec)
    return spec
