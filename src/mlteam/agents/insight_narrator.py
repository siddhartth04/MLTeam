"""Insight Narrator: turn structured evidence into plain-language findings.

One LLM call that reads the evaluation report + explainability report (feature
importances, regression residual slices) and writes a few stakeholder-readable
findings plus honest caveats. It only READS artifacts and produces an
InsightNarrative — it can never change a model or a metric — so on any bad
generation it is skipped loudly rather than allowed to block the run.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from mlteam.contracts import (
    EvaluationReport,
    ExplainabilityReport,
    InsightNarrative,
    ProblemSpec,
    new_artifact,
)
from mlteam.llm.client import LLMClient, LLMValidationError
from mlteam.llm.summaries import evaluation_summary
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)

_MAX_IMPORTANCES = 10

SYSTEM_PROMPT = """You are the Insight Narrator of an autonomous ML engineering \
team. You receive a structured evaluation report and an explainability summary \
(ranked feature importances and, for regression, per-group error slices). Write \
plain-language findings a business stakeholder — not an ML engineer — can act on.

Rules:
- findings: 2-5 short sentences stating what actually drives the target and how \
well the model performs, GROUNDED in the numbers provided. Name the top features \
in plain terms; when residual slices are present, call out the segment where the \
model is weakest. Never invent a feature or a number that is not in the input.
- caveats: 1-3 honest limitations a stakeholder must know before trusting the \
model — e.g. high error on a particular segment, a metric that is only moderately \
above the baseline, or reliance on a feature that may not be available at serving \
time. Do not manufacture caveats; if the evidence is genuinely clean, say the main \
caveat is the usual one (performance is measured on historical data).
- Do NOT recommend actions or next experiments — that is the Critic's job. You \
describe and explain; you do not prescribe.
Return only JSON matching the schema."""


class InsightDraft(BaseModel):
    findings: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


def narrate_insights(
    client: LLMClient,
    store: ArtifactStore,
    *,
    problem: ProblemSpec,
    evaluation: EvaluationReport,
    explainability: ExplainabilityReport,
    run_id: str,
) -> InsightNarrative | None:
    """Draft and store an InsightNarrative; return None (logged) on failure."""
    payload = {
        "task_type": problem.task_type.value,
        "primary_metric": problem.primary_metric,
        "goal_target": problem.target_column,
        "evaluation": evaluation_summary(evaluation),
        "top_features": [
            {"feature": fi.feature, "importance": round(fi.importance, 6)}
            for fi in explainability.importances[:_MAX_IMPORTANCES]
        ],
        "residual_slices": [
            {"feature": s.feature, "value": s.value, "mae": round(s.mae, 6), "n_rows": s.n_rows}
            for s in explainability.residual_slices
        ],
    }
    try:
        result = client.structured(
            agent="insight-narrator",
            system_prompt=SYSTEM_PROMPT,
            user_payload=payload,
            schema=InsightDraft,
        )
    except LLMValidationError as exc:
        _log.warning("insight_narrator_skipped", error=str(exc))
        return None

    narrative = new_artifact(
        InsightNarrative,
        producer_agent="insight-narrator",
        run_id=run_id,
        parent_ids=(evaluation.meta.artifact_id, explainability.meta.artifact_id),
        model_ref=explainability.model_ref,
        findings=tuple(result.output.findings),
        caveats=tuple(result.output.caveats),
    )
    store.put(narrative)
    _log.info(
        "insight_narrative_built",
        findings=len(narrative.findings),
        caveats=len(narrative.caveats),
    )
    return narrative
