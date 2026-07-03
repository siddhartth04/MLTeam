"""Run budget accounting for the iteration loop.

The loop is capped by explicit budgets (iterations, wall time, LLM spend).
Every stop decision is evidence-based and lands in the trace — the loop never
just "runs until it feels done".
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

from mlteam.llm.client import LLMClient

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RunBudget:
    """Hard caps for one run of the iteration loop."""

    max_iterations: int = 3
    max_wall_seconds: float = 1800.0
    max_llm_cost_usd: float = 5.0


class BudgetTracker:
    """Tracks consumption against a RunBudget.

    Wall time comes from a monotonic clock; LLM spend comes straight from the
    client's own accounting (single source of truth for cost).
    """

    def __init__(
        self,
        budget: RunBudget,
        client: LLMClient,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._budget = budget
        self._client = client
        self._clock = clock
        self._started_at = clock()
        self._iterations_used = 0

    @property
    def budget(self) -> RunBudget:
        return self._budget

    @property
    def iterations_used(self) -> int:
        return self._iterations_used

    @property
    def elapsed_seconds(self) -> float:
        return self._clock() - self._started_at

    @property
    def llm_cost_usd(self) -> float:
        return self._client.total_cost_usd

    def record_iteration(self) -> None:
        self._iterations_used += 1

    def can_continue(self) -> tuple[bool, str]:
        """(True, "") while within budget; (False, reason) once exhausted."""
        if self._iterations_used >= self._budget.max_iterations:
            return False, (
                f"iteration budget exhausted "
                f"({self._iterations_used}/{self._budget.max_iterations})"
            )
        if self.elapsed_seconds >= self._budget.max_wall_seconds:
            return False, (
                f"wall-time budget exhausted "
                f"({self.elapsed_seconds:.0f}s/{self._budget.max_wall_seconds:.0f}s)"
            )
        if self.llm_cost_usd >= self._budget.max_llm_cost_usd:
            return False, (
                f"LLM cost budget exhausted "
                f"(${self.llm_cost_usd:.4f}/${self._budget.max_llm_cost_usd:.2f})"
            )
        return True, ""

    def snapshot(self) -> dict[str, Any]:
        """Serializable view for traces and gate summaries."""
        return {
            "iterations_used": self._iterations_used,
            "max_iterations": self._budget.max_iterations,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "max_wall_seconds": self._budget.max_wall_seconds,
            "llm_cost_usd": round(self.llm_cost_usd, 6),
            "max_llm_cost_usd": self._budget.max_llm_cost_usd,
        }
