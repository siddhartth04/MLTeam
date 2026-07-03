"""Budget tracker tests with fake clock and fake cost source."""

from __future__ import annotations

from mlteam.orchestration import BudgetTracker, RunBudget


class FakeCostClient:
    def __init__(self) -> None:
        self.total_cost_usd = 0.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _tracker(**budget_kwargs):
    client = FakeCostClient()
    clock = FakeClock()
    tracker = BudgetTracker(RunBudget(**budget_kwargs), client, clock=clock)
    return tracker, client, clock


def test_within_budget_continues():
    tracker, _, _ = _tracker(max_iterations=3)
    ok, reason = tracker.can_continue()
    assert ok and reason == ""


def test_iteration_budget_exhausts():
    tracker, _, _ = _tracker(max_iterations=2)
    tracker.record_iteration()
    tracker.record_iteration()
    ok, reason = tracker.can_continue()
    assert not ok and "iteration budget" in reason


def test_wall_time_budget_exhausts():
    tracker, _, clock = _tracker(max_wall_seconds=60)
    clock.now += 61
    ok, reason = tracker.can_continue()
    assert not ok and "wall-time budget" in reason


def test_llm_cost_budget_exhausts():
    tracker, client, _ = _tracker(max_llm_cost_usd=0.10)
    client.total_cost_usd = 0.11
    ok, reason = tracker.can_continue()
    assert not ok and "cost budget" in reason


def test_snapshot_is_serializable():
    import json

    tracker, client, clock = _tracker()
    tracker.record_iteration()
    client.total_cost_usd = 0.05
    clock.now += 10
    snap = tracker.snapshot()
    assert json.loads(json.dumps(snap)) == snap
    assert snap["iterations_used"] == 1
    assert snap["elapsed_seconds"] == 10.0
    assert snap["llm_cost_usd"] == 0.05
