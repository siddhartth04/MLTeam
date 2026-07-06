"""API tests: run lifecycle, gates, artifacts, lineage, reports — no live LLM.

Uses a fresh registry per test and drives runs through the recorded fixture, so
nothing touches the network.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlteam.api import registry as registry_module
from mlteam.api.config import APISettings
from mlteam.api.events import RunStatus, Stage
from mlteam.api.registry import RunRegistry

FIXTURE = Path(__file__).parent / "fixtures" / "llm" / "iterative_churn_approved.json"


@pytest.fixture()
def fresh_registry(monkeypatch):
    reg = RunRegistry()
    monkeypatch.setattr(registry_module, "REGISTRY", reg)
    # The app and run_manager import REGISTRY by name; patch those bindings too.
    import mlteam.api.app as app_module
    import mlteam.api.run_manager as rm_module

    monkeypatch.setattr(app_module, "REGISTRY", reg)
    monkeypatch.setattr(rm_module, "REGISTRY", reg)
    return reg


@pytest.fixture()
def client(tmp_path, fresh_registry):
    from mlteam.api.app import create_app

    settings = APISettings(
        store_root=tmp_path / "store",
        upload_dir=tmp_path / "uploads",
        seed_demo=False,
    )
    return TestClient(create_app(settings)), fresh_registry, settings


def _make_csv(path: Path) -> None:
    from synthetic import make_churn_frame

    path.parent.mkdir(parents=True, exist_ok=True)
    make_churn_frame(n_rows=200).write_csv(path)


def _launch(client_tuple, tmp_path) -> str:
    """Seed a completed demo run (auto-approves all gates, synchronous).

    The live launch path blocks at each human gate by design; the demo seeder
    is the correct fully-automated end-to-end path for populated assertions.
    """
    _, reg, settings = client_tuple
    from mlteam.api.demo import DEMO_UI_RUN_ID, seed_demo_run

    seed_demo_run(store_root=settings.store_root, data_dir=settings.upload_dir, registry=reg)
    return DEMO_UI_RUN_ID


def _wait_for_completion(reg: RunRegistry, ui_run_id: str, timeout: float = 120.0) -> None:
    # The seeder is synchronous, so the run is already terminal — assert it.
    terminal = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.REJECTED}
    record = reg.get(ui_run_id)
    assert record is not None and record.status in terminal, (
        f"expected terminal status, got {record.status if record else None}"
    )
    assert record.status is RunStatus.COMPLETED, f"seed run failed: {record.error}"


def test_health(client):
    api, _, _ = client
    assert api.get("/api/health").json() == {"status": "ok"}


def test_full_run_completes_and_populates_api(client, tmp_path):
    api, reg, _ = client
    ui_run_id = _launch(client, tmp_path)
    _wait_for_completion(reg, ui_run_id)

    run = api.get(f"/api/runs/{ui_run_id}").json()
    assert run["status"] == "completed"
    assert run["final_report_id"]
    assert run["llm_cost_usd"] > 0

    # Experiments leaderboard populated.
    exps = api.get(f"/api/runs/{ui_run_id}/experiments").json()
    assert len(exps) >= 2
    assert any(not e["is_baseline"] for e in exps)

    # Lineage graph has nodes and edges.
    lineage = api.get(f"/api/runs/{ui_run_id}/lineage").json()
    assert len(lineage["nodes"]) > 5
    assert len(lineage["edges"]) > 0

    # Problem spec + critique available.
    assert api.get(f"/api/runs/{ui_run_id}/problem-spec").json()["target_column"] == "churned"
    assert api.get(f"/api/runs/{ui_run_id}/critique").json()["verdict"] in (
        "approve",
        "iterate_with_actions",
        "halt",
    )


def test_holdout_artifact_is_protected_in_api(client, tmp_path):
    api, reg, _ = client
    ui_run_id = _launch(client, tmp_path)
    _wait_for_completion(reg, ui_run_id)

    arts = api.get(f"/api/runs/{ui_run_id}/artifacts", params={"type": "DatasetVersion"}).json()
    # Find the holdout dataset (role=holdout) via detail.
    holdout_found = False
    for art in arts:
        detail = api.get(f"/api/artifacts/{art['id']}").json()
        if detail.get("protected"):
            holdout_found = True
            assert "_protected" in detail["payload"]
    assert holdout_found  # the frozen holdout appears but its content is withheld


def test_reports_render_as_html(client, tmp_path):
    api, reg, _ = client
    ui_run_id = _launch(client, tmp_path)
    _wait_for_completion(reg, ui_run_id)
    report_id = api.get(f"/api/runs/{ui_run_id}").json()["final_report_id"]

    business = api.get(f"/api/reports/{report_id}/business")
    technical = api.get(f"/api/reports/{report_id}/technical")
    assert business.status_code == 200
    assert "<!doctype html>" in business.text.lower()
    assert "Holdout" in technical.text


def test_reports_are_downloadable_and_self_contained(client, tmp_path):
    api, reg, _ = client
    ui_run_id = _launch(client, tmp_path)
    _wait_for_completion(reg, ui_run_id)
    report_id = api.get(f"/api/runs/{ui_run_id}").json()["final_report_id"]

    business = api.get(f"/api/reports/{report_id}/business/download")
    assert business.status_code == 200
    assert "attachment" in business.headers["content-disposition"]
    assert f"report-business-{report_id[:12]}.html" in business.headers["content-disposition"]
    # Self-contained: inline CSS + embedded Plotly script, not a broken shell.
    assert "<style" in business.text
    assert "<script" in business.text

    technical = api.get(f"/api/reports/{report_id}/technical/download")
    assert technical.status_code == 200
    assert "attachment" in technical.headers["content-disposition"]


def test_download_unknown_report_is_404(client):
    api, _, _ = client
    assert api.get("/api/reports/does-not-exist/business/download").status_code == 404


def test_download_unknown_kind_is_404(client, tmp_path):
    api, reg, _ = client
    ui_run_id = _launch(client, tmp_path)
    _wait_for_completion(reg, ui_run_id)
    report_id = api.get(f"/api/runs/{ui_run_id}").json()["final_report_id"]
    assert api.get(f"/api/reports/{report_id}/nonsense/download").status_code == 404


def test_gate_approval_flow(client, tmp_path):
    """A run whose gate blocks must surface in /api/gates and resume on approve."""
    import threading

    from mlteam.api.events import StageStatus
    from mlteam.api.registry import GateRequest

    api, reg, settings = client
    ui_run_id = "gate-run"
    reg.create(ui_run_id, goal_text="g", dataset_name="d")

    # Simulate a run thread opening Gate 2 and blocking on it.
    def fake_run():
        req: GateRequest = reg.open_gate(
            ui_run_id, Stage.GATE_PLAN, {"experiments": ["dummy", "lgbm"]}
        )
        req.event.wait(timeout=10)

    t = threading.Thread(target=fake_run, daemon=True)
    t.start()
    time.sleep(0.3)

    gates = api.get("/api/gates").json()
    assert any(g["ui_run_id"] == ui_run_id and g["stage"] == "gate_plan" for g in gates)

    resolved = api.post(f"/api/runs/{ui_run_id}/gate", json={"approved": True, "notes": "ok"})
    assert resolved.json() == {"resolved": True}
    t.join(timeout=5)
    record = reg.get(ui_run_id)
    stage = next(s for s in record.stages if s.stage is Stage.GATE_PLAN)
    assert stage.status is StageStatus.DONE

    # Resolving again is a 409 (no pending gate).
    assert api.post(f"/api/runs/{ui_run_id}/gate", json={"approved": True}).status_code == 409


def test_delete_completed_run(client, tmp_path):
    api, reg, _ = client
    ui_run_id = _launch(client, tmp_path)
    _wait_for_completion(reg, ui_run_id)

    resp = api.delete(f"/api/runs/{ui_run_id}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}
    assert api.get(f"/api/runs/{ui_run_id}").status_code == 404
    assert reg.get(ui_run_id) is None


def test_delete_unknown_run_is_404(client):
    api, _, _ = client
    assert api.delete("/api/runs/does-not-exist").status_code == 404


def test_delete_active_run_is_409(client):
    api, reg, _ = client
    ui_run_id = "in-flight-run"
    reg.create(ui_run_id, goal_text="g", dataset_name="d")
    reg.set_status(ui_run_id, RunStatus.RUNNING)

    resp = api.delete(f"/api/runs/{ui_run_id}")
    assert resp.status_code == 409
    assert reg.get(ui_run_id) is not None  # untouched


def test_run_registry_rename(client):
    _, reg, _ = client
    reg.create("run-x", goal_text="g", dataset_name="d")
    assert reg.rename("run-x", "My churn run") is True
    assert reg.get("run-x").name == "My churn run"


def test_run_registry_rename_unknown_run_returns_false(client):
    _, reg, _ = client
    assert reg.rename("does-not-exist", "name") is False


def test_create_run_with_name_shows_up_in_get(client, tmp_path):
    api, reg, _ = client
    reg.create("run-named", goal_text="predict churn", dataset_name="d.csv", name="Churn v2")
    run = api.get("/api/runs/run-named").json()
    assert run["name"] == "Churn v2"


def test_patch_renames_run(client, tmp_path):
    api, reg, _ = client
    reg.create("run-to-rename", goal_text="predict churn", dataset_name="d.csv")

    patch = api.patch("/api/runs/run-to-rename", json={"name": "Renamed"})
    assert patch.status_code == 200
    assert patch.json()["name"] == "Renamed"
    assert api.get("/api/runs/run-to-rename").json()["name"] == "Renamed"


def test_patch_unknown_run_is_404(client):
    api, _, _ = client
    resp = api.patch("/api/runs/does-not-exist", json={"name": "x"})
    assert resp.status_code == 404


def test_blank_name_falls_back_to_dataset_and_timestamp(client, tmp_path):
    api, reg, _ = client
    reg.create("run-blank-name", goal_text="predict churn", dataset_name="d.csv")
    run = api.get("/api/runs/run-blank-name").json()
    assert run["name"]  # never empty
    assert run["dataset_name"] in run["name"]


def test_explainability_endpoint(client, tmp_path):
    api, reg, _ = client
    ui_run_id = _launch(client, tmp_path)
    _wait_for_completion(reg, ui_run_id)
    explain = api.get(f"/api/runs/{ui_run_id}/explainability").json()
    assert explain is not None
    assert len(explain["importances"]) >= 1
    assert len(explain["figures"]) >= 1
    # Each figure is fetchable as Plotly JSON.
    fig = api.get(f"/api/figures/{explain['figures'][0]['blob_hash']}")
    assert fig.status_code == 200
    assert '"data"' in fig.text
