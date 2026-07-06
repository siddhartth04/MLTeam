# Implementation Plan — Trust & Reasoning Upgrade ("ML OS" phase)

**Audience:** an AI coding agent (Sonnet) executing autonomously, one phase per commit.
**Repo:** `ml-eng-team/` — agentic ML platform (hub-and-spoke agents over a typed artifact store).
**Origin:** principal-architect review on 2026-07-06 after a real user run (Kaggle house prices,
~170k rows) exposed trust gaps. Full context in the "Why" notes per phase.

---

## Ground rules (read first, apply to every phase)

1. **Run order is the priority order.** Phase 0 is mandatory before anything else
   (the test suite is currently RED — commit `d9ccb90` included failing tests).
2. **Every phase ends with:** `uv run ruff check src tests` clean AND
   `uv run pytest -q` fully green, then ONE conventional commit (`feat:`/`fix:`/`test:`),
   then `git push origin main`. If push times out (GitHub is flaky from this machine),
   retry in a background loop with 45s backoff — never skip the push silently.
3. **Never violate these invariants:**
   - The holdout guard: nothing reads holdout artifacts except final reporting.
   - Leakage rules: any op that learns statistics from data must happen inside CV folds
     or be provably row-local if applied pre-split (see `agents/feature_executor.py`
     docstring for the canonical explanation).
   - Determinism: seeds pinned, `n_jobs=1`, no wall-clock/randomness in artifact content.
   - Artifacts are frozen pydantic models; agents are pure functions
     `typed artifacts in → typed artifacts out`; all handoffs via the store.
4. **Fixture replay must stay green.** `tests/fixtures/llm/iterative_churn_approved.json`
   drives the demo + several E2E tests through `FixtureTransport` (sequential replay —
   changing *prompts/payloads* is safe, changing the *number or order of LLM calls* is NOT
   unless you update fixtures).
5. **Do not commit `.env`** (gitignored; contains live Azure credentials).
6. Style: ruff line length 100, type annotations on all signatures, small focused files,
   match surrounding idiom. Comments only for constraints the code can't show.
7. After phases that change runtime behavior, restart the API so the user can test:
   kill the PID on port 8000, then
   `uv run uvicorn mlteam.api.app:app --port 8000` (background), wait for
   `/api/health` to return ok.

### Key file map

| Area | Files |
|---|---|
| Contracts | `src/mlteam/contracts/` (`__init__.py` has `ARTIFACT_REGISTRY` + `__all__`) |
| Agents | `src/mlteam/agents/` (triage, ingest, profiler, splitter, cleaner, feature_engineer, feature_executor, experiment_planner, trainer, requirement, critic, explainer, deployer, time_features) |
| Orchestration | `src/mlteam/orchestration/` (`pipeline.py` = deterministic path + `PipelineResult` + `_train_all`; `agentic.py` = LLM path; `iteration.py` = critique/HPO loop; `gates.py`) |
| Evaluation | `src/mlteam/evaluation/` (metrics.py, report.py) |
| API | `src/mlteam/api/` (`app.py` routes, `views.py` dashboard queries, `run_manager.py` background runs, `registry.py` in-memory run registry, `demo.py` seed) |
| Dashboard | `dashboard/` (Next.js; `app/experiments/page.tsx`, `app/gates/page.tsx`, `lib/api.ts`, `lib/types.ts`) |
| LLM | `src/mlteam/llm/` (`client.py` structured outputs, `schema.py` wire-schema sanitizer — every property must be in `required`, no open dicts) |

---

## Phase 0 — Repair the current red state (BLOCKING)

**Why:** `tests/test_triage.py` was committed with 4 failing tests and lint errors.

Tasks:
1. `tests/test_triage.py` fails because small fixtures trip `MIN_LABELED_ROWS = 50`
   in `src/mlteam/agents/triage.py`. Fix by making every test frame have ≥ 60 labeled
   rows (e.g. repeat patterns with list multiplication) — do NOT lower the production
   threshold; it is a deliberate senior-engineer judgment.
2. Fix lint: import order, remove unused `TriageIssueCode` import, wrap all `E501` lines
   (break the long `triage_dataset(...)` calls into multi-line keyword args).
3. Verify the assertions still test the right behavior:
   - unlabeled rows dropped + counted in `TriageReport`
   - text regression target (`"₹3,500"`, `"1000"`) coerced via `coerce_target_numeric`
   - junk numeric feature (`"> 10"` among ints) coerced via `coerce_column_numeric`
   - too-few-labels raises `ValueError` matching `"too few usable labels"`
   - classification labels that look like sentinels (`"na"`, `"none"`) are PRESERVED
4. Full suite + lint green. Commit `fix: repair triage test suite`.

---

## Phase 1 — Experiment lifecycle + persisted per-experiment metrics (TRUST FIX)

**Why (real incident):** In the user's house-prices run the dashboard showed metrics for
ONE model (MLP) and blanks for LightGBM/XGBoost/CatBoost/dummy, so the user concluded
the others crashed and the winner was picked by default. Reality: all models trained
fine; per-experiment metrics exist only in the in-memory `TrainResult` and the single
persisted `EvaluationReport` covers only best-vs-dummy. `api/views.py::experiments_view`
joins TrainedModel→EvaluationReport, so every non-winner renders `metrics: {}`.
Failures aren't persisted either (`skipped_experiments` is an in-memory tuple), so
"success" and "crash" are visually identical. Fix by persisting an outcome per experiment.

Tasks:
1. New contract `src/mlteam/contracts/outcome.py`:
   ```python
   class ExperimentState(StrEnum):
       COMPLETED = "completed"
       FAILED = "failed"
       SKIPPED = "skipped"

   class ExperimentOutcome(Artifact):
       experiment_id: str
       template_id: str
       state: ExperimentState
       config: dict[str, Any] = Field(default_factory=dict)
       cv_metrics: tuple[MetricSummary, ...] = ()   # reuse the evaluation contract type
       wall_time_seconds: float = 0.0
       error: str | None = None                     # exception text on FAILED
       model_ref: ArtifactRef | None = None         # set when COMPLETED
   ```
   Register in `ARTIFACT_REGISTRY` + `__all__`. Reuse `MetricSummary` from the existing
   evaluation contracts (find it via grep; do not redefine).
2. In `orchestration/pipeline.py::_train_all` (shared by both pipelines): wrap each
   experiment with timing; on success store an `ExperimentOutcome(state=COMPLETED,
   cv_metrics=<from TrainResult.metrics>, model_ref=<TrainedModel ref>)`; on caught
   failure store `state=FAILED, error=str(exc)`. Keep the existing dummy-failure-is-fatal
   rule. `_train_all` will need the `store` (already has it) and `run_id` (already has it).
3. Same persistence in the iteration loop (`orchestration/iteration.py`) for HPO and
   plan-revision experiments — grep for where `train_experiment` is called there.
4. **Quorum policy** in `orchestration/agentic.py` after `_train_all`: let
   `planned = non-dummy specs`, `completed = successful non-dummy`. If
   `len(completed) == 0` → raise `ValueError` listing each failure
   (`"all N candidate experiments failed: <template: error>..."`). If
   `len(completed) < 0.5 * len(planned)` → log a structured warning
   `experiment_quorum_degraded` AND include the failure list in the Gate 2 /
   critique flow (append to the evaluation report's `skipped` visibility — simplest:
   log + keep going; the persisted FAILED outcomes make it visible on the leaderboard).
   Do not silently crown a survivor when everything else burned: the ValueError path
   covers the worst case, the leaderboard covers the rest.
5. Rewrite `api/views.py::experiments_view` to read `ExperimentOutcome` artifacts for
   the run: one row per outcome with `state`, `metrics`, `error`, `wall_time_seconds`.
   Keep the old TrainedModel join as fallback ONLY if a run has zero outcomes
   (backward compat with the seeded demo store — or simpler: reseed demo).
6. Dashboard `app/experiments/page.tsx` + `lib/types.ts`: add `state` and `error`
   columns. FAILED rows render with the existing `bg-danger`-style classes and show
   the error text (truncated, full on hover/expand). COMPLETED rows show metrics.
7. Tests (`tests/test_experiment_outcomes.py`):
   - completed outcome persisted with metrics after a tiny pipeline run
   - a template forced to fail (e.g. spec with an invalid config for a real template)
     produces a FAILED outcome with error text and does not kill the run
   - all-non-dummy-failed raises the quorum ValueError
   - `experiments_view` returns state+error rows
   - update any existing tests that assert on the old leaderboard shape (`tests/test_api.py`).

Commit: `feat: per-experiment lifecycle outcomes + honest leaderboard + quorum policy`.

---

## Phase 2 — Exact-duplicate removal moves PRE-split (LEAKAGE FIX)

**Why (real incident):** the user's run dropped **86,352 duplicate rows** — but the
cleaner runs post-split on train only. Identical rows almost certainly sat in both
train and holdout → holdout metrics inflated by memorization. Exact-dup removal is
row-local-safe pre-split and MUST happen there.

Tasks:
1. Add to `contracts/triage.py`: `TriageIssueCode.DUPLICATE_ROWS`,
   `TriageAction.DROP_DUPLICATE_ROWS`.
2. In `agents/triage.py::triage_dataset`, after the unlabeled-row drop and before the
   feature coercion loop: `n_dupes = len(df) - len(df.unique(maintain_order=True))`;
   if > 0, dedup with `df.unique(maintain_order=True, keep="first")`, record issue +
   remedy with counts. (`maintain_order=True` keeps determinism.)
3. Leave the cleaner's post-split dedup op in place (it becomes a no-op on the main
   path but protects direct callers).
4. Tests in `tests/test_triage.py`:
   - duplicates dropped pre-split with correct counts in the report
   - an E2E through `run_tabular_pipeline` on a frame with planted cross-boundary
     duplicates: assert the train and holdout datasets share zero identical rows
     (read both via the store — holdout is readable in tests? NO — the guard blocks it;
     instead assert `triage_report` counted the dupes and the split input row count
     equals the deduped count).
5. Update the triage module docstring remedy catalog.

Commit: `fix: exact-duplicate removal pre-split — closes train/holdout memorization leak`.

---

## Phase 3 — DecisionRecord: reasoning on the artifacts

**Why:** the system already reasons (profiler leakage warnings, cleaner justifications,
critic verdicts) but decisions surface as bare answers. Every gate should answer WHY
with evidence and alternatives — without inventing fake confidence numbers (explicitly
rejected: LLM self-reported confidence is uncalibrated theater).

Tasks:
1. New `src/mlteam/contracts/decision.py`:
   ```python
   class DecisionRecord(BaseModel):   # frozen
       decision: str
       why: str
       evidence: tuple[str, ...] = ()
       alternatives_considered: tuple[str, ...] = ()
       assumptions: tuple[str, ...] = ()
   ```
2. Add OPTIONAL `reasoning: DecisionRecord | None = None` to `ProblemSpec` and
   `ExperimentPlan` contracts (additive + optional = old stored artifacts still load).
3. `agents/requirement.py`: extend `ProblemSpecDraft` with flat string/list fields
   (`why_target`, `target_alternatives`, `key_assumptions` — REMEMBER the wire-schema
   rules in `llm/schema.py`: no open dicts, every property auto-required). Map them
   into the `DecisionRecord`. Update `SYSTEM_PROMPT` to demand: why this target, which
   other columns were candidates and why rejected, what assumptions were made.
4. `agents/experiment_planner.py`: same treatment — why these templates for THIS data
   profile, which were considered and skipped and why. Additionally extend the plan
   draft with `excluded_templates: list[ExcludedTemplateDraft]` where each entry is
   `{template_id: str, reason: str}` — the prompt must demand coverage of the major
   families not planned (linear, svm/knn, mlp, ensembles, detectors where applicable):
   the plan's NEGATIVE SPACE becomes auditable. Store on the ExperimentPlan contract
   as an optional tuple (additive). The Phase 10 critic reads it.
5. Gate summaries: Gate 1 payload (in `agentic.py`) includes the reasoning fields;
   Gate 2 already carries `rationale` — add `alternatives_considered`.
6. Dashboard `app/gates/page.tsx`: render reasoning blocks (why / evidence /
   alternatives / assumptions) — plain sections, no new dependencies.
7. Fixture caution: the churn fixture's recorded responses will not contain the new
   draft fields → they must be optional-with-defaults in the draft models so replay
   still validates. Assert that in a test.
8. Tests: draft→DecisionRecord mapping; gate payload contains reasoning; fixture
   replay still green.

Commit: `feat: DecisionRecord — every gate answers why, with evidence and alternatives`.

---

## Phase 4 — Profile-lite before Gate 1

**Why:** the requirement agent sees COLUMN NAMES ONLY (`_read_columns`), so it
structurally cannot rank target candidates or warn about leakage at Gate 1. In the
user's run it had to choose between `Price (in rupees)` and `Amount(in rupees)` blind.

Tasks:
1. New `src/mlteam/agents/preview.py`: `column_preview(data_path, n_rows=5000)` —
   deterministic, no artifact, no LLM. Reads a capped sample (`pl.read_csv(...,
   n_rows=5000, infer_schema=False)` then per-column: inferred-ish type via cheap
   parse checks, null fraction, n_unique, up to 3 distinct sample values as strings).
   Handle parquet via existing readers. Must never crash on junk (that is its point).
2. `agentic.py`: replace the bare `columns` payload for `understand_requirements`
   with the preview list. `requirement.py`: pass preview through to the prompt payload,
   update `SYSTEM_PROMPT` to use null-rates/cardinality/samples as evidence for target
   ranking (feeds the Phase 3 DecisionRecord).
3. Keep `_read_columns` for the post-gate validation path (target-exists check).
4. Tests: preview on a messy CSV (nulls, junk, unicode currency) returns sane rows;
   requirement post-validation unchanged; fixture replay unaffected (payload changes
   are safe with sequential FixtureTransport — verify).

Commit: `feat: profile-lite before Gate 1 — evidenced target selection`.

---

## Phase 5 — Deployment dossier (Gate 3 answers "why trust this?")

**Why:** Gate 3 currently shows artifact hashes. All trust evidence already exists as
artifacts; nobody assembles it.

Tasks:
1. `agents/deployer.py::build_deployment_bundle`: accept optional
   `evaluation: EvaluationReport | None`, `explainability=...`, `critique=...`,
   `outcomes: tuple[ExperimentOutcome, ...] = ()` and build the Gate 3 summary dict:
   - comparison table: one row per outcome (template, state, primary-metric mean±std)
   - winner + margin vs runner-up and vs dummy
   - top-5 features by importance
   - critique verdict + any warnings
   - lineage: model / dataset / split artifact ids
   - drift baseline + inference schema summary (already in the bundle)
   - HONESTY NOTE: do not fabricate a "rollback strategy" — there is no model
     registry yet; omit rather than fake.
2. `api/run_manager.py` + `api/demo.py`: pass the new args (they already hold
   `result.final_report`, `explain_report`, `result.final_critique`; outcomes come
   from the store by run_id).
3. Dashboard gate page renders the dossier table.
4. Tests: dossier summary contains comparison rows + winner rationale for a small run.

Commit: `feat: deployment dossier at Gate 3`.

---

## Phase 6 — Residual analysis + insight narrator

**Why:** explainability produces figures but no findings; regression runs lack
residual/error analysis entirely.

Tasks:
1. `agents/explainer.py`: for regression-like tasks add deterministic residual
   analysis on TRAIN data only (residual histogram figure, residuals-vs-predicted
   figure, and a worst-slices table: group by the highest-importance categorical
   feature if one exists, report per-group MAE). Extend `ExplainabilityReport`
   contract additively (optional fields).
2. New `agents/insight_narrator.py` (LLM, one call): input = evaluation summary +
   top importances + residual/slice stats; output `InsightNarrative` artifact
   (registered) with `findings: tuple[str, ...]` (plain-language, evidence-tied) and
   `caveats: tuple[str, ...]`. On `LLMValidationError`: skip loudly (log warning,
   return None) — never block the run.
3. Wire into `api/run_manager.py` after explainability; surface in the report page
   (simplest: include findings in the final report inputs → `reporting/`).
4. Tests: residual fields populated for a regression run; narrator validates against
   a hand-written fixture response; skip-on-invalid path logs and continues.

Commit: `feat: residual analysis + plain-language insight narrator`.

---

## Phase 7 — Outcome ledger (cross-run memory as a PRIOR, never a pruner)

**Why:** the planner starts cold every run. Experience should reorder candidates —
but memory must never delete baselines or skip evaluation (bias fossilization).

Tasks:
1. `src/mlteam/store/ledger.py`: append-only JSONL at `<store_root>/outcome_ledger.jsonl`.
   Record per completed run: `{task_type, n_rows_bucket (log10), n_features,
   categorical_share (rounded to 0.1), winner_template, margin_vs_second,
   primary_metric, timestamp}`. Write from `run_manager` on successful completion
   (and demo seed).
2. `agents/experiment_planner.py`: read up to 5 ledger entries matching
   `task_type` + same `n_rows_bucket` ± 1; include in the LLM payload as
   `"prior_outcomes"`. Add to `SYSTEM_PROMPT`: "priors may REORDER candidates and
   budgets; they may NEVER remove dummy_baseline or lightgbm_default, and every
   planned experiment is still evaluated."
3. The existing plan validator (baselines-first) is the code-level enforcement —
   confirm it rejects plans missing the mandatory baselines regardless of priors.
4. Tests: ledger write/read roundtrip; planner payload includes matching priors;
   validator still rejects baseline-free plans.

Commit: `feat: outcome ledger — cross-run priors for the planner`.

---

## Phase 8 — Fix the cost meter for non-Anthropic models

**Why (real incident):** Azure runs log `unknown_model_pricing model=gpt-4o-2024-11-20`
and the dashboard cost meter reads $0 — a trust meter that lies politely.

Tasks:
1. `src/mlteam/llm/` pricing table (grep `estimate_cost_usd`): add entries for
   `gpt-4o` family (match by prefix so dated snapshots like `gpt-4o-2024-11-20`
   resolve; use $2.50/M input, $10.00/M output).
2. Env override for anything unknown: `MLTEAM_LLM_PRICE_INPUT_PER_M` /
   `MLTEAM_LLM_PRICE_OUTPUT_PER_M` (floats, applied when the table misses).
   Keep the warning when neither table nor override matches.
3. Tests: prefix match works; override works; unknown still returns None + warns.

Commit: `fix: cost tracking for OpenAI/Azure models + price override`.

---

## Phase 9 — Adaptive two-round planning (screen → contend)

**Why:** the planner spends all its intelligence BEFORE any experiment runs — it plans
once, blind to results. A senior engineer runs cheap probes first, reads them, then
spends the budget. This phase makes the initial plan results-conditioned without
unbounded agent loops. It also encodes the diagnostic role of cheap models: linear
regression is an instrument, not a competitor (strong linear ⇒ mostly-linear signal;
linear ≈ dummy while LGBM soars ⇒ nonlinearity justifies ensembles; near-perfect
linear ⇒ LEAKAGE ALARM, flag — never just escalate).

Tasks:
1. In `orchestration/agentic.py`, before the experiment-planning LLM call: run a
   deterministic **screening round** via the existing `_train_all` — fixed specs:
   `dummy_baseline`, `linear_default`, `lightgbm_default` (all defaults, seconds of
   compute). Persist their `ExperimentOutcome`s (Phase 1 machinery) tagged with
   `experiment_id` prefix `screen-`.
2. Pass screening results into `plan_experiments` payload as
   `"screening_results": [{template, primary_metric_mean, primary_metric_std,
   wall_time_seconds}]` alongside the profile and Phase 7 priors.
3. `SYSTEM_PROMPT` additions (evidence-driven selection heuristics):
   - strong linear (close to LGBM) → mostly-linear signal: prefer regularized linear
     + one GBDT; deep stack unlikely to pay for itself.
   - linear ≈ dummy, LGBM strong → nonlinear/interaction-heavy: prioritize GBDT
     family + stacking; MLP only if rows are numerous and dense-numeric.
   - LGBM or linear near-perfect → suspect leakage; say so in the rationale; plan a
     verification (the critic's hard gates will also catch it).
   - skip SVM/KNN above ~20k rows (O(n²)); skip MLP under ~10k rows or with heavy
     categoricals; NEVER skip the mandatory baselines.
4. Round-2 plan must not re-run identical template+config already screened — the plan
   validator should treat screened experiments as satisfying the baselines-first rule
   (reference their outcomes instead of retraining; simplest: keep requiring dummy +
   lightgbm specs in the plan but have `_train_all` reuse an existing outcome when an
   identical template+config+dataset was already trained this run).
5. Gate 2 summary gains the screening table (template, metric, time) so the human sees
   the evidence the plan was conditioned on.
6. Tests: screening outcomes persisted; planner payload contains screening results;
   identical-spec reuse hits (no double training); Gate 2 payload includes the table;
   fixture-replay tests updated (this ADDS no LLM calls — screening is deterministic —
   so replay order is unchanged; verify).

Commit: `feat: adaptive two-round planning — screening evidence conditions the plan`.

---

## Phase 10 — Critic upgrade: full-field evidence + exploration audit

**Why:** the critic exists and is well-designed (report-only input, hard gates that
code-override the LLM, approve/iterate/halt) but it judges ONLY best-vs-dummy. It
cannot compare competitors, audit terminal states, or challenge the plan's negative
space. Upgrade its evidence diet, not its identity. Do NOT add a second reviewer
agent — same LLM, different hat, 2× cost, zero new information.

Tasks:
1. Deterministic pre-checks (new module `evaluation/field_checks.py`, pure functions,
   fed INTO the critique payload — never LLM-computed):
   - terminal-state audit: every spec in the ExperimentPlan (+ screening) has a
     persisted ExperimentOutcome; any hole = structured error (this is a bug trap).
   - failure census: counts by state, failure reasons list.
   - winner margin: (winner − runner-up) on the primary metric in units of pooled
     fold std. Also winner-vs-dummy margin.
   - cost census: total wall time vs Gate-2 estimate.
2. Extend the critique payload in `agents/critic.py` with: the full outcome table
   (template, state, metrics mean±std, time), the field checks above, and the plan's
   `excluded_templates` reasons (Phase 3).
3. `SYSTEM_PROMPT` additions: challenge exploration coverage ("given this profile,
   was excluding family X justified? If not, emit an iterate action to try it");
   when `margin_in_stds < 0.5`, the verdict text MUST address the tie — recommend
   the simpler/cheaper model or justify keeping the winner; explicitly state
   "evidence sufficient for deployment: yes/no, because ...".
4. New HARD gate (code-enforced, `evaluation/` where the existing hard gates live):
   `INCOMPLETE_EXPERIMENT_FIELD` — if the terminal-state audit finds holes, the critic
   cannot approve (this catches orchestration bugs, not model quality).
   Margin-indistinguishability is NOT a hard gate — it demands explanation, not a block.
5. Surface: critique verdict + field checks land in the Gate 3 dossier (Phase 5) and
   the dashboard critique panel.
6. Tests: field checks unit-tested (audit hole detection, margin math with hand
   numbers); critic payload includes outcomes + exclusions; hard gate blocks approval
   on a synthetic hole; fixture updates for the enriched payload (payload changes are
   replay-safe; call count unchanged).

Commit: `feat: critic sees the whole field — outcome audit, margin stats, exploration challenge`.

---

## Explicitly OUT of scope (do not build, even if tempting)

- More agents / new topology — enrich contracts instead (decided in review; this
  includes NOT adding a second reviewer/evaluator agent — Phase 10 upgrades the
  existing critic's evidence instead).
- LLM "confidence scores" — evidence + alternatives only.
- Model registry, rollback automation, monitoring loop — separate future phase.
- Clustering/unsupervised, text/image modalities, distributed training.
- Unbounded bandit-style experiment loops — Phase 9's two bounded rounds + the
  existing budget-capped iteration loop are the sanctioned form of adaptivity.
- Formal significance testing between models — margin-in-fold-stds (Phase 10) is the
  honest check at 3–5 folds; don't dress up low-power tests as rigor.

## Phase dependencies

0 → everything. 1 → 2..10 (outcomes are the substrate). 3 → 9, 10 (DecisionRecord /
exclusions feed them). 9 → 10 (critic audits screening too). 5 consumes 1, 3, 10
outputs as available — build it after 1+3, extend after 10.

## Definition of done (whole plan)

- All 10 phases committed + pushed, suite green, ruff clean after each.
- Demo store reseeded so the dashboard shows the new leaderboard/gate content
  (delete `artifact_store/` + registry state or bump the demo run id).
- API restarted; user can rerun the house-prices dataset and see: triage report,
  honest per-experiment leaderboard incl. failures, reasoned gates, dossier at
  Gate 3, insights in the report, non-zero cost meter.
