"""Deterministic pre-checks fed INTO the Critic's evidence — never LLM-computed.

The Critic used to judge only best-vs-dummy: it could not compare competitors,
audit whether every planned experiment actually reached a terminal state, or
put a number on how close the winner and the runner-up really are. These pure
functions compute that evidence in code, so the LLM interprets facts instead
of being trusted to notice them.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlteam.contracts import ExperimentOutcome, ExperimentSpec, ExperimentState
from mlteam.evaluation.metrics import signed_improvement


@dataclass(frozen=True)
class TerminalStateAudit:
    """Did every planned experiment (including screening) reach a terminal
    state? A hole here is an orchestration BUG, not a model-quality issue —
    it means something crashed hard enough to skip persisting an outcome."""

    planned_ids: tuple[str, ...]
    outcome_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return len(self.missing_ids) == 0


@dataclass(frozen=True)
class FailureCensus:
    completed: int
    failed: int
    skipped: int
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class WinnerMargin:
    """Separation between the winner and its closest competitor, expressed as
    a pooled-sigma EFFECT SIZE (raw gap / sqrt(mean of the two variances)) —
    not just the raw metric gap, and NOT a significance test. A raw gap of
    0.02 means very different things at std=0.001 (huge, real separation)
    versus std=0.05 (statistical noise); the _stds fields make that
    distinction explicit. Always positive when the winner really is better,
    regardless of the metric's own direction. This is the SAME pooled-sigma
    convention BaselineDelta's heuristic fallback uses (sqrt of the MEAN of
    the two variances) — the two numbers are directly comparable. For an
    actual test with a p-value, see BaselineDelta.p_value
    (evaluation/stats.py's paired Nadeau-Bengio test) — this module doesn't
    compute p-values because it only ever sees a single mean+std per
    experiment, not the underlying per-fold sequence."""

    runner_up_template: str | None
    vs_runner_up_raw: float | None
    vs_runner_up_stds: float | None
    vs_dummy_raw: float | None
    vs_dummy_stds: float | None


@dataclass(frozen=True)
class CostCensus:
    total_wall_time_seconds: float
    estimated_total_seconds: float

    @property
    def over_estimate(self) -> bool:
        return self.total_wall_time_seconds > self.estimated_total_seconds


def audit_terminal_states(
    planned_experiment_ids: tuple[str, ...], outcomes: tuple[ExperimentOutcome, ...]
) -> TerminalStateAudit:
    outcome_ids = tuple(o.experiment_id for o in outcomes)
    missing = tuple(eid for eid in planned_experiment_ids if eid not in outcome_ids)
    return TerminalStateAudit(
        planned_ids=planned_experiment_ids, outcome_ids=outcome_ids, missing_ids=missing
    )


def compute_failure_census(outcomes: tuple[ExperimentOutcome, ...]) -> FailureCensus:
    completed = sum(1 for o in outcomes if o.state is ExperimentState.COMPLETED)
    failed = sum(1 for o in outcomes if o.state is ExperimentState.FAILED)
    skipped = sum(1 for o in outcomes if o.state is ExperimentState.SKIPPED)
    reasons = tuple(o.error for o in outcomes if o.state is ExperimentState.FAILED and o.error)
    return FailureCensus(
        completed=completed, failed=failed, skipped=skipped, failure_reasons=reasons
    )


def _metric_mean_std(outcome: ExperimentOutcome, metric: str) -> tuple[float, float] | None:
    m = next((m for m in outcome.cv_metrics if m.name == metric), None)
    return (m.mean, m.std) if m is not None else None


def _margin_in_stds(
    metric: str, winner: tuple[float, float], other: tuple[float, float]
) -> tuple[float, float | None]:
    """(raw_margin, margin_in_pooled_std_units). The stds value is None when
    the pooled std is ~0 (both sides had near-zero fold variance) — a raw gap
    still exists but a "how many stds apart" framing is undefined there."""
    winner_mean, winner_std = winner
    other_mean, other_std = other
    raw = signed_improvement(metric, winner_mean, other_mean)
    pooled_std = ((winner_std**2 + other_std**2) / 2) ** 0.5
    stds = raw / pooled_std if pooled_std > 1e-12 else None
    return raw, stds


def compute_winner_margin(
    outcomes: tuple[ExperimentOutcome, ...],
    primary_metric: str,
    winner_experiment_id: str,
    dummy_experiment_id: str,
) -> WinnerMargin:
    completed = {o.experiment_id: o for o in outcomes if o.state is ExperimentState.COMPLETED}
    winner = completed.get(winner_experiment_id)
    winner_stats = _metric_mean_std(winner, primary_metric) if winner else None
    if winner_stats is None:
        return WinnerMargin(None, None, None, None, None)

    candidates = [
        (eid, o)
        for eid, o in completed.items()
        if eid not in (winner_experiment_id, dummy_experiment_id)
    ]
    scored = [
        (eid, o, _metric_mean_std(o, primary_metric))
        for eid, o in candidates
        if _metric_mean_std(o, primary_metric) is not None
    ]
    # Closest competitor = smallest positive margin over it (the real "how
    # close was this" question — not simply the second-best raw score, though
    # for a correctly-computed ranking they coincide).
    runner_up_template = None
    vs_runner_up_raw = None
    vs_runner_up_stds = None
    if scored:
        margins = [
            (eid, o, *_margin_in_stds(primary_metric, winner_stats, stats))
            for eid, o, stats in scored
        ]
        margins.sort(key=lambda row: row[2])  # smallest raw margin = closest
        _eid, runner_up_outcome, vs_runner_up_raw, vs_runner_up_stds = margins[0]
        runner_up_template = runner_up_outcome.template_id

    vs_dummy_raw = None
    vs_dummy_stds = None
    dummy = completed.get(dummy_experiment_id)
    dummy_stats = _metric_mean_std(dummy, primary_metric) if dummy else None
    if dummy_stats is not None:
        vs_dummy_raw, vs_dummy_stds = _margin_in_stds(primary_metric, winner_stats, dummy_stats)

    return WinnerMargin(
        runner_up_template=runner_up_template,
        vs_runner_up_raw=vs_runner_up_raw,
        vs_runner_up_stds=vs_runner_up_stds,
        vs_dummy_raw=vs_dummy_raw,
        vs_dummy_stds=vs_dummy_stds,
    )


def compute_cost_census(
    outcomes: tuple[ExperimentOutcome, ...], specs: tuple[ExperimentSpec, ...]
) -> CostCensus:
    total_wall_time = sum(o.wall_time_seconds for o in outcomes)
    estimated_total = sum(s.budget.wall_time_seconds for s in specs)
    return CostCensus(
        total_wall_time_seconds=round(total_wall_time, 2),
        estimated_total_seconds=round(estimated_total, 2),
    )
