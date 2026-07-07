"""Task 4 of REPORT_INTEGRITY_ROADMAP.md: the ui_run_id -> pipeline run_id
-> report provenance spine must survive a backend restart.

Each test constructs a FRESH ``RunRegistry`` bound to the same on-disk
``RunStore`` a prior "process" wrote to -- proving persistence actually
round-trips through SQLite rather than merely working within one
in-memory registry instance.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mlteam.api.events import RunStatus, Stage, StageStatus
from mlteam.api.registry import RunRegistry
from mlteam.api.run_store import RunStore


def _fresh_registry(db_path: Path) -> RunRegistry:
    reg = RunRegistry()
    reg.attach_store(RunStore(db_path))
    return reg


def test_completed_run_survives_a_fresh_registry_over_the_same_db(tmp_path):
    db_path = tmp_path / "registry.db"

    first = _fresh_registry(db_path)
    first.create("run-1", goal_text="predict churn", dataset_name="churn.csv", name="Churn v1")
    first.set_status("run-1", RunStatus.RUNNING)
    first.set_result_run_id("run-1", "pipeline-run-abc")
    first.set_cost("run-1", 0.1234)
    first.finish("run-1", status=RunStatus.COMPLETED, final_report_id="report-xyz")

    second = _fresh_registry(db_path)
    second.load_from_store()
    record = second.get("run-1")

    assert record is not None
    assert record.goal_text == "predict churn"
    assert record.dataset_name == "churn.csv"
    assert record.name == "Churn v1"
    assert record.status is RunStatus.COMPLETED
    assert record.result_run_id == "pipeline-run-abc"
    assert record.final_report_id == "report-xyz"
    assert record.llm_cost_usd == 0.1234


def test_in_flight_run_at_load_time_is_marked_failed_interrupted(tmp_path):
    """A run whose background thread died with the old process (no thread
    survives a restart) must not be loaded as QUEUED/RUNNING/
    AWAITING_APPROVAL -- nothing will ever advance it, so the dashboard
    would show a permanently stuck spinner instead of an honest failure."""
    db_path = tmp_path / "registry.db"

    first = _fresh_registry(db_path)
    first.create("run-stuck", goal_text="predict something", dataset_name="d.csv")
    first.set_status("run-stuck", RunStatus.RUNNING)
    # Simulated crash: no finish() call.

    second = _fresh_registry(db_path)
    second.load_from_store()
    record = second.get("run-stuck")

    assert record is not None
    assert record.status is RunStatus.FAILED
    assert record.error is not None
    assert "restart" in record.error


def test_run_awaiting_approval_at_load_time_is_also_marked_failed(tmp_path):
    db_path = tmp_path / "registry.db"

    first = _fresh_registry(db_path)
    first.create("run-gated", goal_text="g", dataset_name="d.csv")
    first.open_gate("run-gated", Stage.GATE_SPEC, summary={"target": "x"})
    assert first.get("run-gated").status is RunStatus.AWAITING_APPROVAL

    second = _fresh_registry(db_path)
    second.load_from_store()
    record = second.get("run-gated")
    assert record.status is RunStatus.FAILED
    assert "restart" in record.error


def test_failed_correction_is_itself_persisted_not_just_in_memory(tmp_path):
    """The FAILED-on-load correction must be written back to the store --
    otherwise every subsequent load would keep re-deriving it from a
    stale RUNNING row, which happens to produce the same answer today but
    is fragile: any future load_from_store change that reads a field other
    than status/error would see the wrong picture."""
    db_path = tmp_path / "registry.db"

    first = _fresh_registry(db_path)
    first.create("run-stuck", goal_text="g", dataset_name="d.csv")
    first.set_status("run-stuck", RunStatus.RUNNING)

    second = _fresh_registry(db_path)
    second.load_from_store()  # corrects run-stuck to FAILED and persists it

    third = _fresh_registry(db_path)
    third.load_from_store()
    record = third.get("run-stuck")
    assert record.status is RunStatus.FAILED
    assert "restart" in record.error


def test_deleted_run_does_not_reappear_after_a_reload(tmp_path):
    db_path = tmp_path / "registry.db"

    first = _fresh_registry(db_path)
    first.create("run-1", goal_text="g", dataset_name="d.csv")
    first.finish("run-1", status=RunStatus.COMPLETED)
    first.delete("run-1")

    second = _fresh_registry(db_path)
    second.load_from_store()
    assert second.get("run-1") is None


def test_renamed_run_keeps_its_new_name_after_a_reload(tmp_path):
    db_path = tmp_path / "registry.db"

    first = _fresh_registry(db_path)
    first.create("run-1", goal_text="g", dataset_name="d.csv", name="Original")
    first.rename("run-1", "Renamed")

    second = _fresh_registry(db_path)
    second.load_from_store()
    assert second.get("run-1").name == "Renamed"


def test_stage_history_survives_a_gate_transition_and_reload(tmp_path):
    """finish()/open_gate()/resolve_gate() persist the stage list (unlike
    update_stage, which is deliberately not persisted on every call --
    see RunRegistry.update_stage's docstring)."""
    db_path = tmp_path / "registry.db"

    first = _fresh_registry(db_path)
    first.create("run-1", goal_text="g", dataset_name="d.csv")
    first.update_stage("run-1", Stage.REQUIREMENT, StageStatus.DONE)
    first.open_gate("run-1", Stage.GATE_SPEC, summary={})
    first.resolve_gate("run-1", approved=True)
    first.finish("run-1", status=RunStatus.COMPLETED)

    second = _fresh_registry(db_path)
    second.load_from_store()
    record = second.get("run-1")
    gate_state = next(s for s in record.stages if s.stage is Stage.GATE_SPEC)
    assert gate_state.status is StageStatus.DONE


def test_bare_registry_with_no_attached_store_works_unpersisted(tmp_path):
    """Most unit tests construct RunRegistry() with no store at all --
    every mutation must remain a pure no-op persistence-wise rather than
    raising, so existing in-memory-only tests are unaffected."""
    reg = RunRegistry()  # no attach_store call
    reg.create("run-1", goal_text="g", dataset_name="d.csv")
    reg.set_status("run-1", RunStatus.RUNNING)
    reg.set_result_run_id("run-1", "pipeline-abc")
    reg.finish("run-1", status=RunStatus.COMPLETED)
    reg.load_from_store()  # also a no-op with no store attached
    assert reg.get("run-1").status is RunStatus.COMPLETED


def test_load_from_store_is_a_no_op_on_an_empty_store(tmp_path):
    db_path = tmp_path / "registry.db"
    reg = _fresh_registry(db_path)
    reg.load_from_store()
    assert reg.list_runs() == []


def test_app_lifespan_wires_attach_store_and_load_from_store(tmp_path, monkeypatch):
    """The API's ``create_app`` lifespan must call ``attach_store`` and
    ``load_from_store`` on REAL startup -- proven by driving the ASGI
    lifespan context directly (``app.router.lifespan_context``) rather
    than through ``TestClient``, since a bare ``TestClient(app)`` does not
    run the lifespan at all in this FastAPI/Starlette version (only
    ``with TestClient(app) as c:`` does).

    Uses ``monkeypatch`` to bind a dedicated ``RunRegistry`` in place of the
    real module-level singleton (the same pattern test_api.py's own
    ``fresh_registry`` fixture uses successfully across ~40 tests) rather
    than mutating the real singleton directly -- an earlier version of this
    test drove the actual ``mlteam.api.registry.REGISTRY`` object and, when
    run in the same session as test_api.py, observed a still-unexplained
    identity drift where ``app_module.REGISTRY`` stopped matching the
    ``registry_module.REGISTRY`` it started equal to. Rather than let a
    cross-test-file pytest/monkeypatch interaction block Task 4, this test
    is written the same proven-reliable way the rest of the suite already
    verifies REGISTRY-touching behavior.
    """
    import mlteam.api.app as app_module
    from mlteam.api.app import create_app
    from mlteam.api.config import APISettings
    from mlteam.api.registry import RunRegistry

    dedicated = RunRegistry()
    monkeypatch.setattr(app_module, "REGISTRY", dedicated)

    settings = APISettings(
        store_root=tmp_path / "store", upload_dir=tmp_path / "uploads", seed_demo=False
    )
    app = create_app(settings)

    async def _run_first_process() -> None:
        async with app.router.lifespan_context(app):
            dedicated.create(
                "run-survives", goal_text="predict churn", dataset_name="churn.csv"
            )
            dedicated.set_result_run_id("run-survives", "pipeline-run-xyz")
            dedicated.finish(
                "run-survives", status=RunStatus.COMPLETED, final_report_id="report-abc"
            )

    asyncio.run(_run_first_process())
    assert (tmp_path / "store" / "registry.db").exists()

    # Simulate a restart: a brand new app instance sharing the SAME
    # dedicated RunRegistry object (as app_module.REGISTRY would be the
    # same object across requests within one real process) but with its
    # in-memory state cleared first, so only load_from_store (not leftover
    # state from the previous block) can explain what's found afterward.
    dedicated._runs.clear()
    dedicated._subscribers.clear()
    second_app = create_app(settings)

    async def _run_second_process() -> dict:
        async with second_app.router.lifespan_context(second_app):
            record = dedicated.get("run-survives")
            assert record is not None
            return {
                "status": record.status,
                "final_report_id": record.final_report_id,
                "goal_text": record.goal_text,
                "result_run_id": record.result_run_id,
            }

    loaded = asyncio.run(_run_second_process())
    assert loaded["status"] is RunStatus.COMPLETED
    assert loaded["final_report_id"] == "report-abc"
    assert loaded["goal_text"] == "predict churn"
    assert loaded["result_run_id"] == "pipeline-run-xyz"


def test_demo_seed_skips_the_pipeline_when_a_completed_row_is_already_loaded(
    tmp_path, monkeypatch
):
    """The actual bug this closes: before persistence, EVERY restart
    re-seeded the demo run (re-running the full pipeline and re-appending
    to the outcome ledger, inflating planner priors toward the demo
    winner). seed_demo_run's existing ``registry.get(DEMO_UI_RUN_ID) is not
    None`` early-return becomes correct across restarts for free once
    load_from_store() populates the registry BEFORE seeding is attempted
    -- this test proves that ordering by making the pipeline call itself
    fail loudly if it is ever reached a second time."""
    import mlteam.orchestration as orchestration_module
    from mlteam.api.demo import DEMO_UI_RUN_ID, seed_demo_run

    store_root = tmp_path / "store"
    data_dir = tmp_path / "uploads"
    db_path = store_root / "registry.db"

    first = _fresh_registry(db_path)
    result = seed_demo_run(store_root=store_root, data_dir=data_dir, registry=first)
    assert result == DEMO_UI_RUN_ID
    assert first.get(DEMO_UI_RUN_ID).status is RunStatus.COMPLETED

    def _must_not_be_called(*_args, **_kwargs):
        raise AssertionError(
            "run_iterative_pipeline was invoked on a second seed attempt -- "
            "the persisted completed-run row should have short-circuited it"
        )

    monkeypatch.setattr(orchestration_module, "run_iterative_pipeline", _must_not_be_called)

    second = _fresh_registry(db_path)
    second.load_from_store()  # mirrors the API lifespan's load-before-seed ordering
    result2 = seed_demo_run(store_root=store_root, data_dir=data_dir, registry=second)
    assert result2 == DEMO_UI_RUN_ID
    assert second.get(DEMO_UI_RUN_ID).status is RunStatus.COMPLETED
