"""DecisionRecord: WHY an agent chose what it chose, not just the choice.

The system already reasons — the profiler flags leakage, the cleaner logs a
justification per op, the critic issues a verdict — but until this contract
existed, a gate only ever saw the bare answer (a target column, a template
list). A human approving that answer had no way to check whether it was a
considered choice or a coin flip, and no way to challenge what was left out.

Deliberately NOT a confidence score: an LLM's self-reported "confidence: 0.87"
is uncalibrated against anything and is theater dressed as rigor. Evidence,
alternatives considered, and stated assumptions are checkable; a number
pulled from the same ungrounded generation process is not.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DecisionRecord(BaseModel):
    """WHY, not just WHAT — attached to a decision-bearing artifact."""

    model_config = ConfigDict(frozen=True)

    decision: str
    why: str = ""
    evidence: tuple[str, ...] = ()
    alternatives_considered: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
