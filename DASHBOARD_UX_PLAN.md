# Dashboard UX Plan — live progress, run names, lineage scoping, downloads

**Audience:** an AI coding agent (Sonnet) executing autonomously, one phase per commit.
**Repo:** `ml-eng-team/` — agentic ML platform. Backend: FastAPI (`src/mlteam/api/`).
Frontend: Next.js dashboard (`dashboard/`).
**Origin:** user testing on 2026-07-06 reported five issues; a principal review traced
each to its root cause (recorded per phase). Two are real backend bugs, not cosmetics.

---

## Ground rules (read first, apply to every phase)

1. **Run order is priority order.** Phase 1 (live progress) is the headline fix and the
   biggest change; Phase 4 (lineage scoping) is a genuine data-correctness bug.
2. **Every phase ends green:** `cd ..; uv run ruff check src tests && uv run pytest -q`
   (from repo root) AND `cd dashboard && npx tsc --noEmit`, then ONE conventional commit
   (`feat:`/`fix:`), then `git push origin main`. If push times out (GitHub is flaky from
   this machine), retry in a background loop with 45s backoff — never skip the push.
3. **Do not break the recorded-fixture tests.** Several E2E tests drive the API/demo
   through `FixtureTransport` replaying `tests/fixtures/llm/*.json` — changing the number
   or order of LLM calls breaks them. None of this plan touches LLM calls, so you must
   NOT introduce any; if a fixture test fails, you changed call flow by mistake.
4. **Restart the API after backend changes** so the user can test: find the PID on port
   8000 (`netstat -ano | grep :8000`), stop it, then
   `uv run uvicorn mlteam.api.app:app --port 8000` in the background; wait for
   `/api/health` to return `{"status":"ok"}`. The Next.js dev server hot-reloads on its
   own — no restart needed for pure frontend changes.
5. **Do not commit `.env`** (gitignored; live Azure credentials).
6. Style: backend ruff line length 100, type hints on all signatures. Frontend: match the
   existing Tailwind + Framer Motion idiom; no new dependencies; run `tsc` before every
   commit. Reference existing components (`Panel`, `StatTile`, `StatusPill`, `AgentFlow`,
   `ActivityFeed`, `EmptyState`) rather than reinventing.

### Key file map

| Area | Files |
|---|---|
| Run manager (background pipeline thread) | `src/mlteam/api/run_manager.py` — `_execute`, `_run_with_stage_tracking`, `RegistryGate` |
| Run registry (live state + SSE fan-out) | `src/mlteam/api/registry.py` — `RunRecord`, `RunRegistry.update_stage/log/set_cost/finish` |
| Pipeline orchestration | `src/mlteam/orchestration/iteration.py` (`run_iterative_pipeline`), `agentic.py` (`run_agentic_pipeline` — the deterministic substrate: ingest→triage→profile→split→clean→features→plan→train→evaluate→critic) |
| Progress event contracts | `src/mlteam/api/events.py` — `Stage`, `StageStatus`, `StageState`, `LogEvent`, `StreamEvent` |
| API routes | `src/mlteam/api/app.py` — `/api/runs`, `/api/runs/{id}/stream` (SSE), `/api/reports/...` |
| Views (store → JSON) | `src/mlteam/api/views.py` — `lineage_graph`, `experiments_view`, etc. |
| Store (content-addressing) | `src/mlteam/contracts/base.py` (`new_artifact` — id excludes run_id), `src/mlteam/store/lineage.py` (`record` — `INSERT OR IGNORE`, first-writer-wins) |
| Dashboard live stream | `dashboard/lib/useRunStream.ts` (SSE hook), `dashboard/app/mission/page.tsx` |
| Dashboard components | `dashboard/components/` — `Shell.tsx` (sidebar + run selector), `AgentFlow.tsx`, `ActivityFeed.tsx`, `RunContext.tsx` |
| Dashboard API client + types | `dashboard/lib/api.ts`, `dashboard/lib/types.ts` |
| Report page | `dashboard/app/report/page.tsx` |

---

## Phase 1 — Live per-stage progress (fixes #1 no live activity + #2 no green signals)

**Why (root cause, traced):** `run_manager._run_with_stage_tracking` marks every
deterministic stage `PENDING`, then calls `fn()` — which runs the ENTIRE pipeline
(`run_iterative_pipeline`) as one blocking call — then flips all stages to `DONE` in a
single batch at the end. So the SSE stream emits nothing between "run started" and "run
finished": no stage goes `active`→`done` as it actually happens, and the activity feed
gets no per-agent log lines. The UI correctly renders whatever it's streamed; it's
streamed nothing mid-run. The frontend (`useRunStream.ts`, `mission/page.tsx`) is already
built to react to `stage`/`log` events live — the backend just never sends them during
the work.

**Fix strategy:** thread a lightweight progress callback from the run manager down through
the pipeline so each agent boundary emits a real stage transition + a log line as it
happens. Do NOT thread it through every agent's signature (invasive); instead pass ONE
optional callback object into `run_iterative_pipeline` → `run_agentic_pipeline`, and call
it at the existing stage boundaries already visible in `agentic.py`.

Tasks:
1. Define a tiny progress protocol in `src/mlteam/orchestration/progress.py` (new):
   ```python
   class ProgressReporter(Protocol):
       def stage(self, stage: str, status: str, detail: str = "") -> None: ...
       def log(self, message: str, *, level: str = "info", stage: str | None = None) -> None: ...
   class NullReporter:  # default — does nothing, so non-API callers are unaffected
       def stage(self, *a, **k): ...
       def log(self, *a, **k): ...
   ```
   Use plain `str` for stage/status/level to avoid a circular import with `api.events`;
   the run manager maps them to the `Stage`/`StageStatus` enums.
2. `run_agentic_pipeline` (agentic.py) and `run_iterative_pipeline` (iteration.py) take
   `reporter: ProgressReporter = NullReporter()`. At each stage boundary that ALREADY
   exists in agentic.py, wrap it: `reporter.stage("ingest","active")` before, and
   `reporter.stage("ingest","done", detail=f"{n_rows} rows")` after — for ingest, triage,
   profile, split, clean, features, plan (gate 2 already surfaces separately), train,
   evaluate, critic. Emit a `reporter.log(...)` line at each (e.g. "Ingested 404 rows, 7
   columns"). Map screening + train to the `TRAIN` stage. Keep messages short and human.
   The iteration loop reports each ITERATE cycle as an `ITERATE` stage detail update.
3. `run_manager._execute`: build a `RegistryReporter` (new small class in run_manager.py)
   that forwards `stage`/`log` to `registry.update_stage` / `registry.log` and refreshes
   `registry.set_cost(client.total_cost_usd)` on every call (so the LLM spend meter ticks
   live too). Pass it into `run_iterative_pipeline`. DELETE the batch-flip loops in
   `_run_with_stage_tracking` (the "mark all PENDING then all DONE" blocks) — the reporter
   now drives real transitions; keep only the REQUIREMENT-done marker before `fn()` and
   the final ITERATE done/skipped marker after (or move those into the reporter path too).
4. The gate stages (`GATE_SPEC`, `GATE_PLAN`, `GATE_DEPLOY`) already transition correctly
   via `RegistryGate` / `open_gate` — do not touch that path; just make sure the
   surrounding non-gate stages now light up too.
5. `demo.py` uses `AutoApproveGate` and its own stage() calls — pass a reporter there too
   (or leave demo's existing explicit stage calls; the demo is synchronous so live
   progress matters less, but keep it consistent).
6. Tests: a new `tests/test_progress_reporting.py` — run `run_agentic_pipeline` with a
   recording reporter (a list-appending stub) over the `agentic_churn_run.json` fixture;
   assert the reporter saw ingest→...→critic each go active then done, in order, and at
   least one log line per stage. Verify no extra LLM calls (fixture still replays).
7. Manual verification after restart: launch a live run, confirm stages turn green
   one-by-one in Mission Control and the activity feed scrolls in real time.

Commit: `feat: live per-stage progress streaming during the run`.

---

## Phase 2 — User-given run names (fixes #3 must remember run ids)

**Why:** runs are identified only by `run-<hex>` everywhere (sidebar, top bar, gate cards).
`RunRecord` has no human-facing name. Users can't tell their runs apart.

Tasks:
1. `api/registry.py`: add `name: str = ""` to `RunRecord`. `RunRegistry.create` takes an
   optional `name`. Add a `rename(run_id, name)` method (guarded by the lock).
2. `api/app.py`: the `POST /api/runs` launch form gains an optional `name: str = Form("")`;
   pass it to `launch_run` → `registry.create`. If blank, default the display name to the
   dataset filename + a short timestamp (compute in the view layer, not stored, so the
   raw name stays truly empty). Add `PATCH /api/runs/{ui_run_id}` accepting `{"name": ...}`
   → `registry.rename`; 404 if unknown. Include `name` in `_run_public`.
3. `api/demo.py`: give the seeded demo run a friendly name ("Demo — churn prediction").
4. Dashboard:
   - `lib/types.ts`: add `name: string` to `RunSummary`.
   - `lib/api.ts`: `launchRun` sends `name`; add `renameRun(uiRunId, name)`.
   - Launch page (`app/page.tsx`): add an optional "Run name" text input above/near the
     goal field.
   - `Shell.tsx` `RunSelector`: show `name || ui_run_id` as the primary label, with the
     `ui_run_id` demoted to a small mono subtitle. Same in `TopBar`.
   - Optional: an inline rename (pencil affordance) in the run selector calling
     `renameRun` then `mutate("/api/runs")`. Keep it simple; a prompt() is acceptable if a
     full inline editor is too much.
5. Tests: `registry.rename` unit test; `POST /api/runs` with a name then `GET /api/runs`
   shows it; `PATCH` renames; blank name still works.

Commit: `feat: user-given run names alongside run ids`.

---

## Phase 3 — Downloadable reports (fixes #5 report should be downloadable)

**Why:** `/api/reports/{id}/business|technical` serve inline HTML (viewed in an iframe /
"open in new tab"), but there is no way to SAVE the report. Users want an artifact they
can keep or share.

Tasks:
1. `api/app.py`: add `GET /api/reports/{report_id}/{kind}/download` (kind ∈
   business|technical) returning the SAME HTML but with
   `Content-Disposition: attachment; filename="report-{kind}-{short_id}.html"`. Reuse the
   existing blob-read logic; the only difference is the header (use `HTMLResponse` with
   `headers={...}` or `Response(media_type="text/html", headers=...)`). The stored report
   HTML is already fully self-contained (inline CSS + embedded Plotly JSON via `<script>`),
   so a saved .html file renders standalone — verify by asserting the business blob
   contains `<script>` (Plotly) and `<style>`/inline css, so a downloaded file is not a
   broken shell.
2. `lib/api.ts`: add `reportDownloadUrl(reportId, kind)`.
3. `app/report/page.tsx`: add a "Download" button next to "Open in new tab" that links to
   the download URL (a plain `<a download>` is enough).
4. Tests: `GET .../download` returns 200, `content-disposition: attachment`, and the body
   contains the report title + `<script` (self-contained).

Commit: `feat: downloadable self-contained HTML reports`.

---

## Phase 4 — Lineage scoped to the actual run (fixes #4 shows every run's artifacts)

**Why (root cause, traced — this is a REAL bug, not cosmetics):** `views.lineage_graph`
DOES filter by `run_id` correctly. The leak is upstream, in content-addressing:
`new_artifact` computes an artifact id from its payload + config hash, deliberately
EXCLUDING `run_id` (so identical work dedupes). `lineage.record` is `INSERT OR IGNORE`, so
the FIRST run to produce a given artifact owns its lineage `run_id` forever; a later run
that produces byte-identical artifacts (same dataset, same seed — very common when a user
re-runs the demo dataset, or the seeded demo re-seeds) does NOT overwrite that row. Result:
the newer run's lineage graph surfaces the OLDER run's artifacts (wrong run_id attribution)
or appears to mix runs. This is the same root cause behind the intermittent critic
terminal-state "hole" seen during phase-10 testing.

**Decide the fix with care — two viable options, pick ONE and note why in the commit:**

- **Option A (recommended, smallest correct fix): a run-scoped membership table.** Keep
  content-addressing and dedup as-is (they are load-bearing for reproducibility). Add a
  many-to-many `run_artifacts(run_id, artifact_id)` table in `store/lineage.py`, written on
  every `record()` (INSERT OR IGNORE on the PAIR, so the same artifact can belong to
  multiple runs). `lineage_graph` and `list_artifacts` filter via THIS table
  (`artifact_id IN (SELECT artifact_id FROM run_artifacts WHERE run_id=?)`) instead of the
  single `artifacts.run_id` column. This correctly shows every artifact a run actually
  used, even shared/deduped ones. The existing `artifacts.run_id` column stays as
  "first producer" provenance.
- **Option B (heavier, do not do unless A is infeasible): include run_id in the artifact
  id.** This breaks dedup and reproducibility guarantees (two identical runs would produce
  different ids) and would ripple through many tests. Rejected unless A can't work.

Tasks (Option A):
1. `store/lineage.py`: add the `run_artifacts` table to `_SCHEMA`; write the pair in
   `record()`. Add `artifact_ids_for_run(run_id) -> tuple[str,...]`. Backfill is
   unnecessary (old rows keep working via the fallback below).
2. `views.lineage_graph` and `views.list_artifacts`: prefer `artifact_ids_for_run`; if it
   returns empty for a run (pre-existing stores with no membership rows), fall back to the
   current `record.run_id == run_id` filter so old runs still render.
3. Tests: two runs over the SAME tiny dataset+seed in ONE store; assert each run's
   `lineage_graph` returns ITS full node set (both runs see the shared deduped artifacts
   under their own run_id), and that run A's graph is not empty because run B created a
   shared artifact first. This is the exact scenario that was broken.
4. Note in the commit which option you chose and why.

Commit: `fix: lineage/artifacts scoped to run via membership table (dedup-safe)`.

---

## Definition of done (whole plan)

- All 4 phases committed + pushed; `uv run pytest -q` + `ruff check` green from repo root,
  `npx tsc --noEmit` clean in `dashboard/`, after each phase.
- API restarted; the user can: launch a named run and watch stages turn green live with a
  scrolling activity feed (Phase 1+2), download a self-contained report (Phase 3), and see
  a lineage graph that contains only that run's artifacts (Phase 4).
- No new LLM calls introduced anywhere (fixture-replay tests still pass unchanged).

## Explicitly OUT of scope

- Reworking the AgentFlow visual design — it already renders live state correctly once
  Phase 1 feeds it real transitions; do not redesign it.
- PDF report export (HTML download is the ask; PDF is a heavier future item).
- Persisting the run registry across API restarts — it is intentionally in-memory; naming
  and live progress do not require persistence.
- Touching the ML pipeline logic, contracts other than the additive registry/lineage
  changes, or the holdout guard.
