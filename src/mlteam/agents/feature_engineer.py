"""Feature Engineering agent: domain-informed feature proposals (whitelisted ops).

The LLM proposes FeatureProposals from the FeatureOpType whitelist; the
deterministic executor (feature_executor.py) then applies the row-local
stateless subset to the train dataset and drops stat-learning ops loudly —
see that module for the exact leakage rules.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from mlteam.contracts import (
    DataProfile,
    FeatureOpType,
    FeaturePlan,
    FeatureProposal,
    ProblemSpec,
    new_artifact,
)
from mlteam.llm.client import LLMClient
from mlteam.llm.summaries import profile_summary
from mlteam.store.interface import ArtifactStore

SYSTEM_PROMPT = """You are the Feature Engineering agent of an autonomous ML \
engineering team. Propose domain-informed derived features for a tabular dataset.

Rules:
- Use ONLY the whitelisted op types in the schema (ratio, difference, product, \
log_transform, binning, datetime_decompose, one_hot_encode, ordinal_encode, \
target_encode, frequency_encode, group_aggregate, text_length).
- PREFER row-local ops: ratio, difference, product, log_transform, \
datetime_decompose, text_length, and binning with explicit params_json \
{"edges": [...]}. The executor runs these. Ops that learn statistics \
(target_encode, frequency_encode, group_aggregate, binning without edges) and \
redundant encodings (one_hot/ordinal — the pipeline already encodes) are \
DROPPED at execution time, so proposing them wastes your feature budget.
- input_columns must be existing dataset columns; never use the target column \
as an input.
- Each proposal needs a domain rationale: WHY this feature should carry signal \
for the stated problem, not just what it computes.
- Prefer a handful of high-conviction features over a long speculative list \
(aim for 3-8).
Return only JSON matching the schema."""


class FeatureProposalDraft(BaseModel):
    name: str
    op_type: FeatureOpType
    input_columns: list[str] = Field(min_length=1)
    # Open dicts cannot ride the structured-output grammar; op parameters
    # travel as a JSON-encoded object string and are parsed + validated here.
    params_json: str = "{}"
    rationale: str

    def parsed_params(self) -> dict[str, Any]:
        parsed = json.loads(self.params_json)
        if not isinstance(parsed, dict):
            raise ValueError("params_json must encode a JSON object")
        return parsed


class FeaturePlanDraft(BaseModel):
    proposals: list[FeatureProposalDraft] = Field(default_factory=list)


def propose_features(
    client: LLMClient,
    store: ArtifactStore,
    *,
    profile: DataProfile,
    problem: ProblemSpec,
    run_id: str,
) -> FeaturePlan:
    """Draft, validate, and store a FeaturePlan."""
    known_columns = {c.name for c in profile.columns}

    def _post_validate(draft: FeaturePlanDraft) -> None:
        names = [p.name for p in draft.proposals]
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        for proposal in draft.proposals:
            try:
                proposal.parsed_params()
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"feature '{proposal.name}': params_json is not a valid JSON object: {exc}"
                ) from exc
            unknown = [c for c in proposal.input_columns if c not in known_columns]
            if unknown:
                raise ValueError(
                    f"feature '{proposal.name}' references unknown columns {unknown}; "
                    f"known columns: {sorted(known_columns)}"
                )
            if problem.target_column in proposal.input_columns:
                raise ValueError(
                    f"feature '{proposal.name}' uses the target column as input — "
                    "that is target leakage"
                )
            if not proposal.rationale.strip():
                raise ValueError(f"feature '{proposal.name}' needs a domain rationale")

    result = client.structured(
        agent="feature-engineer",
        system_prompt=SYSTEM_PROMPT,
        user_payload={
            "problem": {
                "task_type": problem.task_type.value,
                "target_column": problem.target_column,
                "primary_metric": problem.primary_metric,
                "business_constraints": list(problem.business_constraints),
            },
            "profile": profile_summary(profile),
        },
        schema=FeaturePlanDraft,
        post_validate=_post_validate,
    )
    plan = new_artifact(
        FeaturePlan,
        producer_agent="feature-engineer",
        run_id=run_id,
        parent_ids=(profile.meta.artifact_id, problem.meta.artifact_id),
        dataset_ref=profile.dataset_ref,
        proposals=tuple(
            FeatureProposal(
                name=p.name,
                op_type=p.op_type,
                input_columns=tuple(p.input_columns),
                params=p.parsed_params(),
                rationale=p.rationale,
            )
            for p in result.output.proposals
        ),
    )
    store.put(plan)
    return plan
