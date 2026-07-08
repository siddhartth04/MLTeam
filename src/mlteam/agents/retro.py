"""Retro agent: one best-effort LLM call after a run finishes, distilling
0-3 durable lessons from the critique, outcomes, and triage remedies.

The ledger records WHICH template won; nothing before this recorded WHAT
the run taught ("log-transform price-like targets" — the founding
incident's lesson had nowhere to live, so every failure was re-litigated
from scratch). Both approve and halt runs get a retro — halts teach more.

Best-effort, never fails a run: any error (LLM, validation, disk) is
logged and swallowed, same contract as store/ledger.py's outcome write.
Recorded only for live-transport runs (``client.is_live``) — a fixture
replay (including every test and the demo seed) is not a new data point,
the same policy the ledger itself should eventually carry
(REPORT_INTEGRITY_ROADMAP.md Task 7; not landed as of this commit, so this
gate is inline here rather than shared).
"""

from __future__ import annotations

from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from mlteam.contracts import CritiqueReport, ExperimentOutcome, TaskType, TriageRemedy
from mlteam.llm.client import LLMClient
from mlteam.store.lessons import LessonEntry, record_lesson

_log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are the Retro agent of an autonomous ML engineering team. \
A run just finished (approved or halted) -- distill 0-3 DURABLE lessons future \
runs on similar data should know, from the critique, the trained-experiment \
outcomes, and the triage remedies applied to this dataset.

Rules:
- A lesson is a general, imperative proposition ("log-transform price-like \
targets spanning multiple orders of magnitude"), NOT a summary of this run \
("the model achieved roc_auc 0.9"). If nothing generalizes, return zero lessons \
-- an empty list is a valid, often correct answer.
- lesson must be <= 300 characters and stand alone without this run's context.
- evidence must cite the concrete number or finding that grounds the lesson \
(e.g. "MAE was 32% of median target value with no transform applied").
- Only lessons a SENIOR engineer would actually act on next time — not vague \
advice ("try more models") and not something already true by construction \
(e.g. do not "recommend" leakage checks, they always run).
Return only JSON matching the schema."""


class LessonDraft(BaseModel):
    lesson: str = Field(min_length=1, max_length=300)
    evidence: str = Field(min_length=1)


class RetroDraft(BaseModel):
    lessons: list[LessonDraft] = Field(default_factory=list, max_length=3)


def _outcomes_summary(outcomes: tuple[ExperimentOutcome, ...]) -> list[dict[str, object]]:
    return [
        {
            "experiment_id": o.experiment_id,
            "template_id": o.template_id,
            "state": o.state.value,
            "metrics": {m.name: round(m.mean, 6) for m in o.cv_metrics},
            "error": o.error,
        }
        for o in outcomes
    ]


def run_retro(
    client: LLMClient,
    store_root: Path,
    *,
    run_id: str,
    task_type: TaskType,
    critique: CritiqueReport,
    outcomes: tuple[ExperimentOutcome, ...] = (),
    triage_remedies: tuple[TriageRemedy, ...] = (),
) -> tuple[LessonEntry, ...]:
    """Distill and record 0-3 lessons from this run. Never raises."""
    if not client.is_live:
        return ()
    try:
        result = client.structured(
            agent="retro",
            system_prompt=SYSTEM_PROMPT,
            user_payload={
                "task_type": task_type.value,
                "verdict": critique.verdict.value,
                "findings": [
                    {
                        "severity": f.severity.value,
                        "category": f.category,
                        "evidence": f.evidence,
                        "recommendation": f.recommendation,
                    }
                    for f in critique.findings
                ],
                "iteration_actions": list(critique.iteration_actions),
                "outcomes": _outcomes_summary(outcomes),
                "triage_remedies": [
                    {
                        "action": r.action.value,
                        "column": r.column,
                        "justification": r.justification,
                    }
                    for r in triage_remedies
                ],
            },
            schema=RetroDraft,
        )
        recorded = tuple(
            record_lesson(
                store_root,
                lesson=draft.lesson,
                evidence=draft.evidence,
                task_type=task_type.value,
                source_run_id=run_id,
            )
            for draft in result.output.lessons
        )
        return recorded
    except Exception:  # retro is a nice-to-have; never breaks a completed run
        _log.warning("retro_skipped", run_id=run_id, exc_info=True)
        return ()
