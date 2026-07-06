"""FastAPI application backing the dashboard.

Read routes over the artifact store + lineage + traces, a live SSE stream of run
progress, a launch endpoint (upload a CSV + goal), and the three human-gate
actions. All heavy compute happens on the run manager's background threads.
"""

from __future__ import annotations

import json
import queue
import time
import uuid
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from mlteam.api.config import APISettings, get_settings
from mlteam.api.demo import seed_demo_run
from mlteam.api.registry import REGISTRY, RunActiveError, RunRecord
from mlteam.api.run_manager import launch_run
from mlteam.api.views import (
    artifact_detail,
    critique_view,
    experiments_view,
    explainability_view,
    final_report_view,
    lineage_graph,
    list_artifacts,
    problem_spec_view,
)
from mlteam.llm.transport import AnthropicTransport, FixtureTransport
from mlteam.observability.logging import configure_logging
from mlteam.store import open_store

_log = structlog.get_logger(__name__)

_DEMO_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "llm"
    / "iterative_churn_approved.json"
)


class GateAction(BaseModel):
    approved: bool
    notes: str = ""


class RenameRequest(BaseModel):
    name: str


def _store(settings: APISettings) -> Any:
    return open_store(settings.store_root)


def _display_name(record: RunRecord) -> str:
    if record.name:
        return record.name
    stamp = record.created_at.strftime("%b %d, %H:%M")
    return f"{record.dataset_name} — {stamp}"


def _run_public(record: RunRecord) -> dict[str, Any]:
    return {
        "ui_run_id": record.run_id,
        "name": _display_name(record),
        "goal_text": record.goal_text,
        "dataset_name": record.dataset_name,
        "status": record.status.value,
        "created_at": record.created_at.isoformat(),
        "result_run_id": record.result_run_id,
        "final_report_id": record.final_report_id,
        "llm_cost_usd": record.llm_cost_usd,
        "error": record.error,
        "stages": [s.model_dump(mode="json") for s in record.stages],
    }


def create_app(settings: APISettings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(json_output=False)
    app = FastAPI(title="mlteam Mission Control API", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        if settings.seed_demo:
            seed_demo_run(store_root=settings.store_root, data_dir=settings.upload_dir)

    # -- health ----------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # -- runs ------------------------------------------------------------------

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return [_run_public(r) for r in REGISTRY.list_runs()]

    @app.get("/api/runs/{ui_run_id}")
    def run(ui_run_id: str) -> dict[str, Any]:
        record = REGISTRY.get(ui_run_id)
        if record is None:
            raise HTTPException(404, "run not found")
        return _run_public(record)

    @app.delete("/api/runs/{ui_run_id}")
    def delete_run(ui_run_id: str) -> dict[str, bool]:
        try:
            deleted = REGISTRY.delete(ui_run_id)
        except RunActiveError as exc:
            raise HTTPException(409, str(exc)) from exc
        if not deleted:
            raise HTTPException(404, "run not found")
        return {"deleted": True}

    @app.patch("/api/runs/{ui_run_id}")
    def rename_run(ui_run_id: str, body: RenameRequest) -> dict[str, Any]:
        if REGISTRY.get(ui_run_id) is None:
            raise HTTPException(404, "run not found")
        REGISTRY.rename(ui_run_id, body.name)
        return _run_public(REGISTRY.get(ui_run_id))

    @app.get("/api/runs/{ui_run_id}/events")
    def run_events(ui_run_id: str, limit: int = Query(500, le=2000)) -> list[dict[str, Any]]:
        record = REGISTRY.get(ui_run_id)
        if record is None:
            raise HTTPException(404, "run not found")
        return [e.model_dump(mode="json") for e in list(record.events)[-limit:]]

    @app.get("/api/runs/{ui_run_id}/stream")
    def run_stream(ui_run_id: str) -> StreamingResponse:
        if REGISTRY.get(ui_run_id) is None:
            raise HTTPException(404, "run not found")
        return StreamingResponse(
            _sse(ui_run_id), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    # -- launch ----------------------------------------------------------------

    @app.post("/api/runs")
    async def create_run(
        goal_text: str = Form(...),
        demo: bool = Form(False),
        name: str = Form(""),
        file: UploadFile | None = File(None),
    ) -> dict[str, str]:
        ui_run_id = f"run-{uuid.uuid4().hex[:10]}"
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        if file is not None:
            data_path = settings.upload_dir / f"{ui_run_id}-{file.filename}"
            data_path.write_bytes(await file.read())
            dataset_name = file.filename or "uploaded.csv"
        else:
            from mlteam.api.demo import _demo_dataset

            data_path = settings.upload_dir / f"{ui_run_id}-demo.csv"
            _demo_dataset(data_path)
            dataset_name = "demo_churn.csv"

        # Demo mode (or no credentials configured) replays the recorded
        # fixture; live mode uses LiteLLM if a model is configured (any
        # provider), else the direct Anthropic transport.
        transport, model = _select_transport(demo, settings)
        launch_run(
            ui_run_id=ui_run_id,
            data_path=data_path,
            goal_text=goal_text,
            dataset_name=dataset_name,
            transport=transport,
            store_root=settings.store_root,
            model=model,
            name=name,
        )
        return {"ui_run_id": ui_run_id}

    # -- gates -----------------------------------------------------------------

    @app.get("/api/gates")
    def gates() -> list[dict[str, Any]]:
        return [
            {"ui_run_id": rid, "stage": g.stage.value, "summary": g.summary}
            for rid, g in REGISTRY.pending_gates()
        ]

    @app.post("/api/runs/{ui_run_id}/gate")
    def resolve_gate(ui_run_id: str, action: GateAction) -> dict[str, bool]:
        ok = REGISTRY.resolve_gate(ui_run_id, action.approved, action.notes)
        if not ok:
            raise HTTPException(409, "no pending gate for this run")
        return {"resolved": True}

    # -- artifacts & lineage ---------------------------------------------------

    @app.get("/api/runs/{ui_run_id}/artifacts")
    def run_artifacts(ui_run_id: str, type: str | None = None) -> list[dict[str, Any]]:
        result_run_id = _result_run_id(ui_run_id)
        return list_artifacts(_store(settings), run_id=result_run_id, artifact_type=type)

    @app.get("/api/artifacts/{artifact_id}")
    def artifact(artifact_id: str) -> dict[str, Any]:
        detail = artifact_detail(_store(settings), artifact_id)
        if detail is None:
            raise HTTPException(404, "artifact not found")
        return detail

    @app.get("/api/runs/{ui_run_id}/lineage")
    def run_lineage(ui_run_id: str) -> dict[str, Any]:
        return lineage_graph(_store(settings), run_id=_result_run_id(ui_run_id))

    # -- experiments, spec, critique, explain ----------------------------------

    @app.get("/api/runs/{ui_run_id}/experiments")
    def experiments(ui_run_id: str) -> list[dict[str, Any]]:
        rid = _result_run_id(ui_run_id)
        return experiments_view(_store(settings), rid) if rid else []

    @app.get("/api/runs/{ui_run_id}/problem-spec")
    def problem_spec(ui_run_id: str) -> dict[str, Any] | None:
        rid = _result_run_id(ui_run_id)
        return problem_spec_view(_store(settings), rid) if rid else None

    @app.get("/api/runs/{ui_run_id}/critique")
    def critique(ui_run_id: str) -> dict[str, Any] | None:
        rid = _result_run_id(ui_run_id)
        return critique_view(_store(settings), rid) if rid else None

    @app.get("/api/runs/{ui_run_id}/explainability")
    def explainability(ui_run_id: str) -> dict[str, Any] | None:
        rid = _result_run_id(ui_run_id)
        return explainability_view(_store(settings), rid) if rid else None

    # -- reports & figures -----------------------------------------------------

    @app.get("/api/reports/{report_id}/business", response_class=HTMLResponse)
    def business_report(report_id: str) -> HTMLResponse:
        report = final_report_view(_store(settings), report_id)
        if report is None:
            raise HTTPException(404, "report not found")
        html = _store(settings).blob_path(report.business_report_blob).read_text("utf-8")
        return HTMLResponse(html)

    @app.get("/api/reports/{report_id}/technical", response_class=HTMLResponse)
    def technical_report(report_id: str) -> HTMLResponse:
        report = final_report_view(_store(settings), report_id)
        if report is None:
            raise HTTPException(404, "report not found")
        html = _store(settings).blob_path(report.technical_report_blob).read_text("utf-8")
        return HTMLResponse(html)

    @app.get("/api/figures/{blob_hash}", response_class=PlainTextResponse)
    def figure(blob_hash: str) -> PlainTextResponse:
        try:
            data = _store(settings).blob_path(blob_hash).read_text("utf-8")
        except Exception as exc:
            raise HTTPException(404, "figure not found") from exc
        return PlainTextResponse(data, media_type="application/json")

    return app


def _has_api_key() -> bool:
    import os

    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _select_transport(demo: bool, settings: APISettings) -> tuple[Any, str]:
    """Pick the run's transport + model string.

    Precedence: explicit demo flag > no credentials at all > configured
    LiteLLM model (any provider) > default AnthropicTransport.
    """
    if demo or (not settings.llm_model and not _has_api_key()):
        return FixtureTransport.from_file(_DEMO_FIXTURE), settings.anthropic_model
    if settings.llm_model:
        from mlteam.llm.litellm_transport import LiteLLMTransport

        kwargs: dict[str, Any] = {}
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key
        if settings.llm_api_base:
            kwargs["api_base"] = settings.llm_api_base
        return LiteLLMTransport(**kwargs), settings.llm_model
    return AnthropicTransport(), settings.anthropic_model


def _result_run_id(ui_run_id: str) -> str | None:
    record = REGISTRY.get(ui_run_id)
    return record.result_run_id if record else None


def _sse(ui_run_id: str) -> Any:
    q = REGISTRY.subscribe(ui_run_id)
    try:
        while True:
            try:
                event = q.get(timeout=15.0)
            except queue.Empty:
                yield "event: heartbeat\ndata: {}\n\n"
                continue
            yield f"event: {event.type}\ndata: {json.dumps(event.data, default=str)}\n\n"
            if event.type == "done":
                # Give the client a moment, then close.
                time.sleep(0.1)
                break
    finally:
        REGISTRY.unsubscribe(ui_run_id, q)


app = create_app()
