# Report Integrity Roadmap

Implementation plan from the 2026-07-07 report-pipeline audit. Execute tasks **in order**, one
conventional commit per task. Each task is self-contained: a fresh session can pick up any task
using only this file plus the referenced source files.

## Background — why this roadmap exists (read once)

During a house-prices regression run (`ui run-4e069fa1e1`, pipeline `run-1336f204e5b7`), the user
saw classification content (ROC-AUC, PR-AUC, churn, support calls, contract type) attributed to
their regression experiment. Forensics established:

- **The report assembly is NOT the bug.** The run's own report blobs were verified clean
  (rmse/mae/r2 only, correct title, correct verdict). `ReportInputs` → `build_final_report`
  consumed only this run's artifacts.
- **Verified leak #1:** `GET /api/runs/{ui_run_id}/artifacts` and `/lineage` resolve
  `result_run_id` via the in-memory registry ([src/mlteam/api/app.py](src/mlteam/api/app.py)
  `_result_run_id`). That value is `None` for a run's entire lifetime (it is set only seconds
  before completion in [src/mlteam/api/run_manager.py](src/mlteam/api/run_manager.py)) and `None`
  for unknown ids. `list_artifacts(store, run_id=None)`
  ([src/mlteam/api/views.py](src/mlteam/api/views.py)) means "no filter" — the endpoint returns
  the **entire store across all runs**. Empirically reproduced: a nonexistent run id returned 222
  artifacts spanning 6+ pipeline runs, including the demo churn run's artifacts. This is the
  vocabulary the user saw.
- **Verified leak #2:** the run registry (`REGISTRY` in
  [src/mlteam/api/registry.py](src/mlteam/api/registry.py)) is process memory. Every backend
  restart erases all run records (the ui_run_id → pipeline run_id → final_report_id provenance
  spine, plus `goal_text`, which exists nowhere else) while the artifact store persists. The demo
  churn run is re-seeded on each restart and becomes the only populated run in the sidebar.
- **Latent contamination path:** Live mode without an API key silently falls back to replaying
  the recorded churn LLM fixture against any dataset (`_select_transport` in app.py).
- **Governance gap (observed live):** critic verdict `halt` does not stop the pipeline — Gate 3
  deployment was offered and approved for a model whose every candidate had R² < 0.
- **Learning-state contamination:** `seed_demo_run` ([src/mlteam/api/demo.py](src/mlteam/api/demo.py))
  appends to the outcome ledger (`record_outcome` in
  [src/mlteam/store/ledger.py](src/mlteam/store/ledger.py) — append-only, no idempotency key) on
  every restart, inflating planner priors toward the demo winner.

**What must NOT be broken** (verified strengths): the holdout guard — including
`GuardedArtifactStore.blob_path` which runs `_check` — content-addressed lineage, Jinja
autoescape (with the single documented `Markup` exemption for Plotly JSON), the cleaning/feature
replay on the holdout, seeded bootstrap CIs, and fixture-replay demo mode itself.

## Conventions (apply to every task)

### Engineering standard — non-negotiable

Write every change as a senior engineer shipping to production, not as a patch that makes tests
pass. Concretely, in this codebase that means:

- **Fail loud, never silently degrade.** No bare `except Exception: pass`. Catch the narrowest
  exception you can name, and either handle it meaningfully or let it propagate to the layer
  that reports it (`run_manager`'s failure handler, FastAPI's error responses). The root cause
  of this entire roadmap was silent-wrong-answer behavior — do not add more of it.
- **Validate at boundaries.** Everything crossing a trust boundary (HTTP params, upload
  filenames, LLM output, rows read from SQLite) is validated before use, with an error message
  that tells the caller what to fix. Internal call chains rely on types, not re-validation.
- **Docstrings explain *why*, not what.** Match the existing style: every non-trivial function
  states the constraint or failure mode that justifies its existence (see
  `_sanitize_upload_filename` or `_sse` in app.py for the house standard). No comments that
  narrate the next line.
- **Full type annotations** on every new function signature, `from __future__ import
  annotations` at module top, pydantic/dataclass models over loose dicts for anything that
  crosses a module boundary.
- **Small and immutable.** Functions under ~50 lines; new files focused (one concern per
  module); contracts are frozen — return new objects, never mutate a stored artifact or a
  shared record outside the registry's lock.
- **No magic values.** Thresholds, filenames, SQL, and env-var names live in named module-level
  constants or `APISettings`, with a comment stating why the value was chosen.
- **Concurrency discipline.** The registry is shared across threads: any new mutable state goes
  under its existing `RLock`; SQLite access from multiple threads uses one connection per
  operation or `check_same_thread=False` with the lock held — pick one and document it.
- **Logging via structlog** with structured key-value fields (`_log.info("event_name", k=v)`),
  never f-string prose logs or `print`.
- **Tests are behavior specs.** Each test asserts one observable behavior with a name that
  reads as a claim (`test_delete_rejects_active_run_until_cancelled`). Test the failure paths —
  the 404s, the raised contract errors — not just the happy path. No test that merely mirrors
  the implementation.
- **Leave the codebase better.** If a task touches a function with a latent bug or a stale
  docstring, fix it in the same commit and mention it in the commit body — but do not refactor
  beyond the task's blast radius.

### Workflow

- Match surrounding style exactly (naming, import grouping, error patterns) — the reviewer
  should not be able to tell which lines are new from style alone.
- Tests first where practical. Every task lists its required tests.
- Verification gate before every commit, from `ml-eng-team/`:
  1. `uv run pytest -q` — full suite green (slow; run in background).
  2. `uv run ruff check src tests` — clean.
  3. If dashboard files changed: `npx tsc --noEmit` in `dashboard/` — clean.
- One commit per task: `<type>: <description>` (feat/fix/refactor/test/chore), body explaining
  the *why*, then push to `origin main`.
- Contract (pydantic) changes must be backward-compatible: new fields optional with defaults so
  previously stored artifacts still deserialize.
- Do not touch the holdout guard semantics. Do not remove demo mode.

---

## Task 1 — Kill the unscoped store fallback (S0, ship first)

**Problem.** `run_artifacts` and `run_lineage` in [src/mlteam/api/app.py](src/mlteam/api/app.py)
pass `_result_run_id(ui_run_id)` (often `None`) into `list_artifacts` / `lineage_graph`, which
treat `None` as "return everything".

**Change.**
1. In both endpoints: if `REGISTRY.get(ui_run_id)` is `None` → `HTTPException(404, "run not found")`.
2. If the record exists but `result_run_id` is `None` (run in flight) → return an explicit empty
   payload, never a global scan:
   - artifacts: `[]`
   - lineage: `{"nodes": [], "edges": [], "pending": True}`
3. Make the misuse impossible to reintroduce: change `list_artifacts` and `lineage_graph`
   signatures in [src/mlteam/api/views.py](src/mlteam/api/views.py) so `run_id: str` is
   **required** (no `None` default). Grep for all callers and update. If any caller genuinely
   needs a whole-store listing (none should from API routes), it must call a separately named
   `list_all_artifacts` so the intent is explicit.
4. Apply the same 404-on-unknown-run check to `/experiments`, `/problem-spec`, `/critique`,
   `/explainability` (they currently return `[]`/`None` for unknown ids — keep `[]`/`None` only
   for the known-but-in-flight case).

**Tests** (extend `tests/test_api.py`):
- `GET /api/runs/ghost/artifacts` → 404 (this currently returns the whole store — the regression
  test that matters most).
- Registered run with `result_run_id=None` → artifacts `[]`, lineage has `pending: true`, and
  crucially contains zero artifacts from other runs even when the store holds several.
- Completed run → scoped listing unchanged.

**Dashboard check:** lineage/trace pages must tolerate the `pending` field and 404s (SWR error →
existing error/empty states). Verify with `tsc`; adjust `lib/types.ts` if `LineageGraph` needs
the optional `pending?: boolean`.

Commit: `fix(api): 404 unknown runs and never fall back to unscoped store queries`

---

## Task 2 — `ReportInputs.validate()`: the task/metric/run coherence contract (S1)

**Problem.** [src/mlteam/reporting/builder.py](src/mlteam/reporting/builder.py) trusts callers to
assemble a coherent `ReportInputs`. Nothing verifies inputs share a run, that the evaluation
belongs to the model, or that metric names are legal for the task type. Assembly is duplicated in
`run_manager.py` and `demo.py` and can drift.

**Change.**
1. New module `src/mlteam/reporting/contract.py`:
   - `class ReportContractError(ValueError)` — message must name every violation found (collect
     all, then raise once; don't stop at the first).
   - `ALLOWED_METRICS: dict[TaskType, frozenset[str]]` derived **from
     `scoring_for(task_type)`** ([src/mlteam/evaluation/metrics.py](src/mlteam/evaluation/metrics.py))
     — single source of truth, never a hand-copied list.
   - `validate_report_inputs(inputs: ReportInputs) -> None` enforcing:
     a. All present artifacts agree on `meta.run_id`: model, evaluation, critique, split_plan,
        and (when not None) explainability, insights, feature_plan, cleaning_manifest.
     b. `inputs.evaluation.model_ref.artifact_id == inputs.model.meta.artifact_id`.
     c. Every name in `evaluation.cv_metrics` and every `baseline_deltas[*].metric` ∈
        `ALLOWED_METRICS[task_type]`.
     d. `inputs.problem.primary_metric` ∈ `ALLOWED_METRICS[task_type]`.
     e. Classification-only sections are absent for regression/forecasting:
        `evaluation.calibration.expected_calibration_error is None` and
        `evaluation.decision_threshold is None`; symmetric check that regression metrics never
        appear for classification tasks.
2. Call it as the **first line** of `build_final_report`. A violation must fail the run loudly
   (the existing `except Exception` in run_manager already surfaces it as FAILED with the trace).

**Tests** (new `tests/test_report_contract.py`): build a minimal valid `ReportInputs` fixture,
then break it one axis at a time — mixed run_ids, evaluation pointing at a different model id, a
`roc_auc` cv metric on a regression spec, `rmse` on a classification spec, calibration present
for regression, illegal primary metric — each raises `ReportContractError` naming the axis; the
valid fixture passes. Also assert `build_final_report` invokes validation (monkeypatch or an
end-to-end negative case).

Commit: `feat(reporting): fail-loud ReportInputs contract (run coherence + task-metric legality)`

---

## Task 3 — Post-render vocabulary linter (S1, defense-in-depth)

**Problem.** Even with Task 2, future template edits or narrative fields could leak wrong-task
vocabulary. A rendered-output check catches the whole class regardless of cause.

**Change.** In `src/mlteam/reporting/contract.py`:
- `lint_report_html(html: str, task_type: TaskType, primary_metric: str) -> None`.
- Forbidden set = `union(ALLOWED_METRICS.values()) - ALLOWED_METRICS[task_type]`, matched as
  whole words, case-insensitive (`\b{token}\b`). **Note:** match metric-name tokens only
  (`roc_auc`, `pr_auc`, `f1_macro`, `log_loss`, `accuracy`, `rmse`, `mae`, `r2`, ...), never
  prose words like "calibration" — the technical template legitimately renders a Calibration
  section header with "Not measured for this run." for regression.
- Require `primary_metric` to appear at least once.
- Call for both business and technical HTML in `build_final_report`, after rendering, before
  `put_blob`. Raise `ReportContractError` on failure.

**Tests:** regression HTML containing `roc_auc` fails; classification HTML containing `rmse`
fails; the real demo-fixture pipeline (existing phase-5 e2e test) still passes — that test is the
proof the linter has no false positives on genuine reports.

Commit: `feat(reporting): post-render task-vocabulary linter on generated reports`

---

## Task 4 — Persistent run registry (S0, the provenance spine)

**Problem.** All `RunRecord`s live in process memory. Restart destroys ui↔pipeline↔report
mappings and `goal_text` (which exists nowhere else and is rendered into the report), orphans
completed runs, and triggers demo re-seeding.

**Change.**
1. New `src/mlteam/api/run_store.py`: SQLite persistence at `store_root / "registry.db"` (its own
   file; do not touch guard.db). One table `runs` with columns: `ui_run_id` (PK), `name`,
   `goal_text`, `dataset_name`, `status`, `created_at`, `result_run_id`, `final_report_id`,
   `error`, `llm_cost_usd`, `stages_json`. Events are ephemeral — do NOT persist the event deque.
2. `RunRegistry` gains an optional `run_store` and becomes write-through: `create`, `finish`,
   `rename`, `delete`, `set_status`, `set_cost`, and stage updates persist the row (stage updates
   may batch: persisting on `finish`/gate transitions is acceptable; document the choice).
3. On API startup (in `create_app` lifespan, before demo seeding): load rows into the registry.
   Any loaded run whose status is `queued`/`running`/`awaiting_approval` → mark
   `FAILED, error="interrupted by server restart"` (its thread is gone; be honest about it).
4. Demo seeding now consults the persisted record: if a completed `demo-seed` row exists, skip
   the pipeline entirely (this also stops per-restart store/ledger churn).
5. Wire-up: registry construction currently happens at module import (`REGISTRY = RunRegistry()`).
   Keep the singleton but add `REGISTRY.attach_store(path)` called from the lifespan — avoids
   import-order problems in tests.

**Tests** (new `tests/test_run_store.py` + extend `test_api.py`): create+finish a run against a
tmp store; construct a **fresh** registry over the same db → record present with
`final_report_id` and `goal_text`; in-flight run at load time → FAILED/interrupted; delete
removes the row; demo seed skipped when row exists (assert `seed_demo_run` returns early —
monkeypatch the pipeline to fail if called).

Commit: `feat(api): persist run registry to SQLite — provenance survives restarts`

---

## Task 5 — Reports that identify themselves + run-scoped report route (S2)

**Problem.** The report HTML names only the internal pipeline run id; no dataset identity, no
task type in the business report; the dashboard fetches reports by bare content-addressed id, so
nothing ever cross-checks report↔run. Misattribution is undetectable by inspection.

**Change.**
1. Contracts ([src/mlteam/contracts/](src/mlteam/contracts/)): `FinalReport` gains optional
   `task_type: str = ""`, `dataset_name: str = ""`, `ui_run_id: str = ""` (defaults keep old
   artifacts deserializable). Populate in `build_final_report` — add `ui_run_id: str = ""` and
   `dataset_name: str = ""` to `ReportInputs`, plumbed from run_manager (which knows both) and
   demo.py.
2. Templates ([src/mlteam/reporting/templates.py](src/mlteam/reporting/templates.py)): both
   reports get a provenance block directly under the `<h1>`: dashboard run id + pipeline run id,
   dataset name, task type, primary metric, seed, env hash, generated_at. The business report
   currently has NO provenance — this is the fix for "which dataset is this report about?".
3. New run-scoped routes in app.py: `GET /api/runs/{ui_run_id}/report/{kind}` (and `/download`)
   that resolve `final_report_id` **from the run record** — the single source of truth — then
   serve the blob. 404 if the run has no report. Keep the existing by-id routes for direct links,
   but the dashboard switches to the run-scoped ones
   ([dashboard/lib/api.ts](dashboard/lib/api.ts) `reportUrl`/`reportDownloadUrl` take `runId`,
   [dashboard/app/report/page.tsx](dashboard/app/report/page.tsx) updated accordingly).
   Download filename becomes `report-{kind}-{ui_run_id}.html`.

**Tests:** FinalReport round-trips new fields; rendered HTML contains both ids + dataset name
(extend the phase-5 report assertions); new route serves the right blob and 404s when
`final_report_id` is null; old stored FinalReport (fields absent) still renders via by-id route.

Commit: `feat(reporting): provenance headers + run-scoped report routes`

---

## Task 6 — No silent fixture fallback in Live mode (S1)

**Problem.** `_select_transport` (app.py): Live mode without credentials silently replays the
recorded churn fixture against any dataset — recorded churn reasoning stitched into arbitrary
runs.

**Change.**
1. In `create_run`: if `demo=False` and no credentials and no `settings.llm_model` →
   `HTTPException(400, "Live mode requires an API key or configured llm_model — switch to Demo (replay) or set ANTHROPIC_API_KEY")`.
   `_select_transport` keeps fixture fallback ONLY for `demo=True`.
2. `RunRecord` + `_run_public` gain `transport: str` (`"live"` / `"fixture-replay"`), set at
   launch; persist it (Task 4 column — add in this commit via small migration:
   `ALTER TABLE ... ADD COLUMN` guarded by a PRAGMA check).
3. Dashboard: show a small "replay" badge next to runs with `transport == "fixture-replay"`
   (sidebar + TopBar; [dashboard/components/Shell.tsx](dashboard/components/Shell.tsx),
   `lib/types.ts`).

**Tests:** launch live-without-key → 400 (monkeypatch env to clear keys); demo launch still
works; `transport` appears in `/api/runs` payload.

Commit: `fix(api): hard-fail live mode without credentials; label fixture-replay runs`

---

## Task 7 — Demo stops polluting the learning ledger (S2)

**Problem.** `seed_demo_run` calls `_record_ledger_outcome` every seed; `record_outcome` is
append-only with no idempotency → planner priors inflate toward the demo winner with every
restart (pre-Task-4 restarts already wrote duplicates).

**Change.**
1. Remove the `_record_ledger_outcome` call from `seed_demo_run`. Policy: **priors learn from
   live runs only.**
2. In `run_manager._execute`, skip the ledger write when the run's transport is fixture replay
   (pass a `record_ledger: bool` through `launch_run`, set from the transport type in
   `create_run`).
3. One-off hygiene note in the commit body: existing duplicate demo rows in `ledger.jsonl` may be
   deleted manually; do not write migration code for it.

**Tests:** seed demo twice against a tmp store → zero ledger rows; a (mocked-fixture) launched
run with `record_ledger=False` writes none; live path still writes.

Commit: `fix(demo): fixture-replay runs never write the outcome ledger`

---

## Task 8 — `halt` verdict blocks deployment (S1, governance)

**Problem.** run_manager proceeds to explain→deploy(Gate 3)→report regardless of critic verdict.
Observed live: halt at 16:32, deployment gate offered and approved at 16:40, for a model with
negative R².

**Change.** In `run_manager._execute` (and mirrored in `demo.py`, though the demo verdict is
approve): when `result.final_critique.verdict` is `HALT`:
- Still run explainability and the final report (the report is the evidence trail).
- Skip `build_deployment_bundle` / Gate 3 entirely: stage `DEPLOY → SKIPPED,
  "blocked: critic verdict halt"`, log at warning level.
- Override hook: `APISettings.allow_deploy_on_halt: bool = False` (env
  `MLTEAM_ALLOW_DEPLOY_ON_HALT`). When true, Gate 3 is offered but its summary dict MUST carry
  `"critic_verdict": "halt"` and a top-level warning line so the human approver acknowledges it.

**Tests:** pipeline result with halt verdict → no DeploymentBundle artifact, DEPLOY stage
skipped, report still built; with override setting → gate requested and summary contains the halt
warning.

Commit: `feat(orchestration): critic halt blocks Gate 3 unless explicitly overridden`

---

## Task 9 — Endpoint hygiene batch (S3)

**Changes** (one commit):
1. `/api/figures/{blob_hash}` (app.py): serve only blobs referenced by some
   `ExplainabilityReport.figures[*].blob_hash` or `FinalReport.figure_blobs` — build the allowlist
   per request by scanning those two artifact types (they are few); 404 otherwise. (The holdout
   guard already protects holdout blobs — this closes "serve any unprotected blob as JSON".)
2. Immutable-content cache headers on `/api/reports/*` and `/api/figures/*`:
   `Cache-Control: public, max-age=31536000, immutable` (safe: content-addressed).
3. Drift uploads: after `evidently_drift_report` completes, delete the temp upload file
   (`finally:` block) instead of accumulating in `upload_dir`.

**Tests:** report blob hash via figures endpoint → 404; genuine figure hash → 200 with cache
header; drift endpoint leaves no file behind (tmp upload_dir listing before/after).

Commit: `fix(api): figure-blob allowlist, immutable cache headers, drift upload cleanup`

---

## Task 10 — Cancellation checkpoints inside the run (UX correctness)

**Problem.** `registry.cancel()` exists but is only checked before and after
`run_iterative_pipeline` — a multi-hour training stage is uncancellable in practice.

**Change.**
1. New `RunCancelledError(RuntimeError)` in `src/mlteam/orchestration/` (e.g. `budget.py` or a
   tiny `cancellation.py`).
2. Optional `should_stop: Callable[[], bool] | None` threaded through
   `run_iterative_pipeline` → `run_tabular_pipeline` → the per-experiment training loop in
   [src/mlteam/orchestration/pipeline.py](src/mlteam/orchestration/pipeline.py) (and the
   iteration loop in the agentic orchestrator). Check **between experiments** and **between
   iterations** — never mid-fit (a partial fit is not a consistent artifact). On true → raise
   `RunCancelledError`.
3. `run_manager`: pass `lambda: registry.is_cancelled(ui_run_id)`; catch `RunCancelledError` →
   `finish(status=CANCELLED)` (replaces the current string-match on `RuntimeError("run cancelled")`
   — keep that branch working for the pre-pipeline check or migrate it to the new exception).

**Tests:** launch a run against a `should_stop` that flips true after the first experiment →
pipeline raises `RunCancelledError`, at least one but not all experiments trained; run_manager
path yields CANCELLED status (existing cancel tests extended).

Commit: `feat(orchestration): cancellation checkpoints between experiments and iterations`

---

## Task 11 — Reports tell the data story (stakeholder fit)

**Problem.** Gate 2's triage evidence (e.g. "9,684 rows dropped: 'Call for Price' ×9684") and the
full experiment leaderboard (including failures like a skipped CatBoost) appear in no report —
the two things a stakeholder and a reviewer respectively ask for first.

**Change.**
1. `ReportInputs` gains `triage: TriageReport | None = None` and
   `outcomes: tuple[ExperimentOutcome, ...] = ()` — plumbed from run_manager/demo (outcomes are
   already fetched there via `list_experiment_outcomes`).
2. Business template: new "Data preparation" section — rows before/after, each remedy as a plain
   sentence (action, column, affected rows, justification).
3. Technical template: "Experiment leaderboard" table — every outcome (id, template, state,
   primary-metric mean±std, wall time, error for failed/skipped rows).
4. Update the vocabulary linter fixtures if new section text introduces metric tokens (it should
   not — table cells use each run's own metrics, which are legal by Task 2).

**Tests:** phase-5 e2e report contains the triage section and a leaderboard row for every
outcome, including a synthetic skipped one.

Commit: `feat(reporting): triage data-story and full experiment leaderboard in reports`

---

## Task 12 (stretch) — `mlteam report rebuild <pipeline-run-id>`

CLI that reconstructs `ReportInputs` purely from the store (scoped fetches by run id; goal_text
and ui ids from the Task-4 registry db) and regenerates both HTMLs. Acceptance: byte-identical
output modulo `generated_at` for a completed run in a cold process. Proves "reproducible solely
from the run ID" end to end. Defer until Tasks 1–5 are merged.

---

## Deferred by design (do not attempt as side effects)

- **RunScope façade** (store handle hard-bound to one run id, view layer accepts only it): the
  right end-state for making unscoped queries unrepresentable, but it touches every view and
  agent callsite. Task 1's required-argument change captures most of the value; do the façade as
  its own reviewed effort.
- Multi-tenant auth on report routes; separate demo store root; O(store) view-scan indexing.
  Note them in commit bodies where adjacent, but do not implement here.
