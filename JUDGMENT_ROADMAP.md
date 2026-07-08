# Judgment Roadmap — from evaluation harness to engineering teammate

Implementation plan from the 2026-07-08 Staff-engineer evaluation. Execute tasks **in order**,
one conventional commit per task. Each task is self-contained: a fresh session can pick up any
task using only this file plus the referenced source files. This roadmap is about **judgment
capabilities**; `REPORT_INTEGRITY_ROADMAP.md` (tasks 5–12 still open there) is about integrity
plumbing — they can proceed in parallel, but do not duplicate work: its Task 8 is absorbed as
Task 0 here.

## Background — the evaluation verdict this roadmap answers (read once)

MLTeam's custody/verification half (holdout guard, leakage battery, Nadeau-Bengio paired
significance, OOF discipline, content-addressed provenance) is production-trustworthy. Its
decision-making half is not. The evidence, from two live runs on 187,531 rows of real house-price
data (2026-07-08, `run-dac59ad719`):

- Target spanned ₹100K–₹14B (four orders of magnitude, violently right-skewed). The system
  trained every model on raw rupees. **No code path exists to transform a target.** Result:
  r² = 0.51, RMSE ≈ 6× MAE (outlier-dominated squared error — the textbook signature of a
  target needing `log`). A first-year Kaggle competitor makes this move reflexively; the
  platform *cannot*.
- The critic said **approve** at r² = 0.51, because its bar is "significantly beats the dummy"
  (dummy r² ≈ 0). Nothing captures what error the business can tolerate, so nothing can judge
  fit-for-purpose.
- The iteration budget was spent on another LightGBM config (`exp-lgbm-rev1`) — model-space
  search — when the single highest-value move was problem-space (transform the target).
  Residual slices are computed and consumed by nothing.
- ~1 hour of statistically rigorous permutation importance was computed for a model that
  plausibly shouldn't ship. No component asks "is this stage still worth its cost?"
- The lesson of the whole incident ("price-like targets need log transform") has **nowhere to
  live**: the outcome ledger stores winner-template priors keyed to dataset shape, not
  propositions. The platform cannot learn it.

The five ranked blockers: (1) action-space ceiling, (2) no fit-for-purpose judgment/refusal,
(3) no evidence-driven iteration, (4) no learning substrate, (5) no value-of-computation
reasoning. Tasks below map to them.

**Current state to verify before starting** (things already fixed on 2026-07-08, commits
`a0276fa`..`d937896` — do not re-implement): per-experiment/per-fold progress logging via
`ProgressReporter` threaded through `_train_all` and `compute_oof`; `_calibration_and_threshold`
skips the OOF refit for non-classification tasks (`is_classification` guard); the pipeline run id
is minted upfront in `run_manager._execute` and registered via `set_result_run_id` at launch;
`lint_rendered_metric_names` checks structured metric names, never prose.

## Conventions (apply to every task)

### Engineering standard — non-negotiable

Write every change as a senior engineer shipping to production, not as a patch that makes tests
pass:

- **Fail loud, never silently degrade.** No bare `except Exception: pass`. Catch the narrowest
  exception you can name; handle it meaningfully or let it propagate to the layer that reports
  it. Silent-wrong-answer behavior is the root cause of most incidents in this codebase's
  history.
- **Validate at boundaries** (HTTP params, LLM output, file content, rows from SQLite), with
  error messages that tell the caller what to fix. Internal call chains rely on types.
- **Docstrings explain WHY, not what.** House standard: every non-trivial function states the
  constraint or failure mode that justifies its existence (see `_sanitize_upload_filename` in
  `src/mlteam/api/app.py` or `compute_oof` in `src/mlteam/evaluation/oof.py`).
- **Full type annotations**, `from __future__ import annotations` at module top,
  pydantic/dataclass models over loose dicts across module boundaries.
- **Small and immutable.** Functions under ~50 lines; contracts are frozen — new objects, never
  mutation of stored artifacts.
- **No magic values.** Thresholds and names live in named module-level constants with a comment
  stating why the value was chosen.
- **Logging via structlog** structured key-value events, never f-string prose or `print`.
- **Tests are behavior specs** with names that read as claims; test the failure paths, not just
  the happy path. No test that merely mirrors the implementation.
- **Backward-compatible contracts**: new pydantic fields optional with defaults so previously
  stored artifacts still deserialize.
- **Circular-import discipline**: `mlteam.orchestration`'s import chain loops back into
  `agents`/`evaluation`/`reporting`. For type-only needs use `TYPE_CHECKING` imports (see
  `evaluation/oof.py` for the established pattern); for runtime needs use function-local imports
  with a comment naming the cycle.

### Workflow

- Verification gate before every commit, from `ml-eng-team/`:
  1. `uv run pytest -q` — full suite green (slow; run in background).
  2. `uv run ruff check src tests` — clean.
  3. If dashboard files changed: `npx tsc --noEmit` in `dashboard/` — clean.
- One commit per task: `<type>: <description>`, body explaining the WHY (match the tone/detail
  of commits `c4f0192` and `d937896` — read `git log -3` first). Do NOT push (the user pushes
  manually). Do not restart backend/dashboard without checking no run is in flight
  (`curl -s localhost:8000/api/runs` — no `"running"` status).
- Do not touch holdout-guard semantics. Do not remove demo mode or human gates.

---

## Task 0 — Critic `halt` blocks deployment (trust floor)

This is `REPORT_INTEGRITY_ROADMAP.md` Task 8, absorbed here because nothing else in this roadmap
is credible while a `halt` verdict still leads to a deployment gate being offered. **Observed
live**: halt at 16:32 → Gate 3 offered and approved at 16:40 → report built, for a model whose
every candidate had negative R².

**Change** (in `src/mlteam/api/run_manager.py::_execute`, mirrored check in `demo.py` though the
demo verdict is approve):
- When `result.final_critique.verdict is Verdict.HALT`: still run explainability (cheap form —
  see Task 6 once it lands; until then, run as-is) and still build the final report (the report
  is the evidence trail). Skip `build_deployment_bundle` / Gate 3 entirely: stage
  `DEPLOY → SKIPPED, "blocked: critic verdict halt"`, `reporter.log` at warning level.
- Override hook: `APISettings.allow_deploy_on_halt: bool = False` (env
  `MLTEAM_API_ALLOW_DEPLOY_ON_HALT` — check `config.py` for the env-prefix convention). When
  true, Gate 3 IS offered but its summary dict must carry `"critic_verdict": "halt"` and a
  top-line warning so the human approver acknowledges what they are signing.

**Tests** (`tests/test_api.py` or a new `tests/test_halt_gating.py`): pipeline result with halt
verdict → no DeploymentBundle artifact for the run, DEPLOY stage skipped with the blocking
detail, FinalReport still built; with the override setting → gate requested and its summary
contains the halt warning. Build the halt result by monkeypatching the critic or using a
fixture transport that returns a halt critique — see `FixtureTransport.from_texts` usage in
`tests/test_phase5_pipeline.py` for the pattern.

Commit: `feat(orchestration): critic halt blocks Gate 3 unless explicitly overridden`

---

## Task 1 — Judgment trap-suite benchmark (measure before improving)

**Problem.** "Behaves like a senior engineer" is currently unfalsifiable. Without a benchmark of
judgment traps, every capability added later cannot be shown to work, and existing strengths
(leakage battery, triage) have no regression protection as capabilities are added.

**Change.**
1. New `benchmarks/judgment_traps.py`: deterministic generators (seeded numpy/polars), each
   returning `(DataFrame, TrapSpec)` where `TrapSpec` is a small frozen dataclass:
   `name`, `target_column`, `task_type`, `expected` (one-line description of the senior call),
   `blocked_by` (task number in this roadmap, or `None` if it must pass today).
2. New `tests/test_judgment_traps.py`: one test per trap, running the **deterministic** pipeline
   (`run_tabular_pipeline` — no LLM, no fixtures needed) against the generated data and
   asserting the trap-specific behavior. Traps whose capability doesn't exist yet are marked
   `@pytest.mark.xfail(strict=True, reason="blocked by JUDGMENT_ROADMAP Task N")` — **strict**,
   so when the capability lands, the xfail turns into an error until the marker is removed in
   that task's commit (this is deliberate: the benchmark enforces that capability tasks flip
   their traps).

Initial traps (8 — keep generators under ~30 lines each):
- `skewed_price_target` (n≈3000): lognormal target spanning ≥3 orders of magnitude, a few
  informative features. Assert (xfail, Task 2): winning `TrainedModel.target_transform ==
  "log1p"` and its primary-metric mean beats an identically-configured raw-target run of the
  same generator (run both inside the test; the comparison is the point).
- `leaky_feature`: one feature = target + tiny noise. Must PASS today: assert the
  `near_perfect_features` leakage check exists in the EvaluationReport and did not pass, OR the
  cleaner dropped the column (either defense counts — assert at least one fired).
- `id_memorization`: unique string id column. Must PASS today: cleaning manifest contains a
  `drop_column` op for it.
- `duplicate_rows`: 20% exact duplicates. Must PASS today: triage remedy
  `drop_duplicate_rows` recorded, rows_after < rows_before.
- `placeholder_labels`: numeric target where 10% of values are the string "Call for Price".
  Must PASS today (regression-protects the 2026-07-07 triage fix): triage remedies contain
  `coerce_target_numeric` then `drop_unlabeled_rows`, and the coercion justification names the
  placeholder string.
- `high_cardinality_categorical` (n≈4000): a 150-level categorical carrying most of the signal.
  Assert (xfail, Task 5): primary metric beats a threshold the generator constructs to be
  reachable only with usable encoding of that column (calibrate the threshold while writing the
  generator: run once with the column dropped to find the no-signal floor, set threshold
  midway to the with-signal ceiling; hardcode both with a comment showing the calibration).
- `temporal_drift_disguised`: rows time-ordered with target drift and a date column, task
  presented as plain REGRESSION. Assert (xfail, no task yet — reason "future: adversarial
  validation / temporal-risk detection"): some artifact warns that random CV overestimates.
- `constant_target`: zero-variance target. Assert: the pipeline fails LOUDLY with a message
  naming the target column (verify what actually happens first — if it currently fails with an
  obscure sklearn error deep in training, add a triage/profiler check that raises a clear
  `ValueError` early, in this same task; that check is small and belongs with the trap).
3. New `benchmarks/run_judgment.py`: runs all traps, prints a scorecard (name | expected |
   status: PASS / FAIL / NOT-YET(Task N)) and exits non-zero on any FAIL that isn't an expected
   xfail. This is the human-readable view; pytest is the enforcement.

**Tests:** the suite IS the tests. Also assert the four must-pass traps in a plain (non-xfail)
run so today's strengths are locked in.

Commit: `feat(benchmarks): judgment trap suite -- senior-engineer calls as executable specs`

---

## Task 2 — Target transformation verb (the proven miss)

**Problem.** No code path can transform a regression target. The 2026-07-08 run is the proof of
cost. This is the highest value-per-line change available in the platform.

**Design decisions (already made — do not re-litigate):**
- Mechanism: `sklearn.compose.TransformedTargetRegressor` wrapping the template-built pipeline,
  with `func=np.log1p, inverse_func=np.expm1`. This keeps every scorer, CV metric, holdout
  metric, and baseline delta in ORIGINAL units automatically (predict() inverse-transforms), so
  comparability against the raw-target dummy floor is preserved with zero metric plumbing.
- Whitelist, not free-form: supported transforms = `{"log1p"}` initially. A module-level
  constant with a docstring explaining log1p over log (tolerates exact zeros; targets must be
  ≥ 0, enforced loudly).
- Scope: `TaskType.REGRESSION` only. FORECASTING is excluded (lag/rolling features are computed
  from the raw target pre-split; mixing raw-scale features with transformed-scale fitting is a
  correctness question deferred deliberately — say so in the docstring).

**Change.**
1. Contracts: `ExperimentSpec` and `TrainedModel` gain `target_transform: str | None = None`
   (optional, backward compatible; INCLUDED in hashing — a transformed model is a different
   model). Grep `class ExperimentSpec` / `class TrainedModel` in `src/mlteam/contracts/`.
2. New helper in `src/mlteam/agents/trainer.py` (or `templates/registry.py` — put it beside
   `get_template`): `wrap_target_transform(pipeline, transform: str | None, task_type)` —
   returns the pipeline unchanged for `None`; raises `ValueError` for an unknown transform name
   or a non-regression task; wraps in `TransformedTargetRegressor` for `"log1p"`.
3. Apply it at EVERY pipeline-construction site, or the transform silently vanishes on rebuild
   paths — this is the critical correctness point. Grep for `get_template(` and
   `.build(` across `src/`: at minimum `train_experiment` (trainer), `compute_oof`
   (evaluation/oof.py — add `target_transform: str | None = None` param, threaded from
   `model.target_transform` by both of its callers), and the HPO retrain path (grep
   `src/mlteam/orchestration/hpo.py` / `iteration.py` for where tuned configs rebuild). The
   serialized joblib pipeline already IS the wrapped object, so `builder.py`'s holdout scoring
   and serving need no changes — state that in the commit body.
4. Trigger — profiler recommendation: in `src/mlteam/agents/profiler.py`, when the TARGET column
   is numeric with `min >= 0` and (skewness > 2.0 or p99/p50 > 20 — constants with rationale
   comments; compute from the sampled column it already profiles), add a `ProfileWarning` (or
   equivalent existing warning structure) with code `target_transform_recommended`, detail
   naming the evidence ("target spans X orders of magnitude, skew=Y").
5. Consumption:
   - Deterministic path (`pipeline.py::run_tabular_pipeline` and `default_experiment_specs`):
     when the train-profile carries the recommendation, set `target_transform="log1p"` on every
     regression spec (dummy included — a dummy wrapped in log1p predicts the geometric mean, a
     legitimate floor).
   - Agentic path: `plan_experiments`' prompt (grep `src/mlteam/agents/experiment_planner.py`)
     gains the recommendation in its payload and instructions that it MAY set
     `target_transform` per experiment; the wire schema for `ExperimentSpec` must accept the new
     optional field (check the sanitizer's additionalProperties handling); validation rejects
     values outside the whitelist LOUDLY (drop the spec with a logged warning, per the existing
     malformed-proposal convention in `feature_executor.py`).
6. Reporting: technical report's problem table (templates.py) gains a "Target transform" row
   when the winner has one, stating metrics are reported in original units.

**Tests** (new `tests/test_target_transform.py`):
- `wrap_target_transform` unit behavior incl. loud failures (unknown name, classification task,
  negative targets → the fit itself should raise; assert the error is comprehensible).
- End-to-end on the Task 1 `skewed_price_target` generator: deterministic pipeline with the
  recommendation → winner has `target_transform == "log1p"`, all reported metrics are
  original-scale (rmse within sane bounds of raw-target magnitudes, NOT log-scale ~0–10 —
  assert order of magnitude), and the transformed run's primary metric beats the raw run's.
- OOF consistency: `compute_oof` with `target_transform="log1p"` produces predictions in
  ORIGINAL units (assert scale), so permutation importance and residual slices stay honest.
- Flip `skewed_price_target`'s xfail marker (strict xfail will force this).

Commit: `feat(pipeline): log1p target transformation -- first problem-space verb`

---

## Task 3 — Fit-for-purpose: capture the decision, raise the critic's bar

**Problem.** The critic approved r² = 0.51 / MAE ≈ 32% of the median price because "beats dummy
significantly" is its only quantitative bar. Nothing captures what the model is FOR.

**Change.**
1. Contracts: `ProblemSpec` gains optional fields (defaults keep old artifacts loading):
   `decision_use: str = ""` (what decision the prediction powers),
   `error_tolerance_note: str = ""` (the user's stated tolerance, free text),
   `human_baseline_note: str = ""` (what happens today without the model).
2. Requirement agent (`src/mlteam/agents/requirement.py`): prompt extracts these from the goal
   text when present, leaves them empty when absent — NEVER invents them (explicit instruction;
   an invented tolerance is worse than none).
3. Gate 1 surfaces the gaps: the gate summary dict lists any of the three that are empty as
   explicit open questions ("Error tolerance: NOT PROVIDED — the critic cannot judge
   fit-for-purpose without it"). The human sees exactly what the system does not know.
4. Deterministic evidence for regression: in `src/mlteam/agents/critic.py`'s
   `compute_quantitative_gates` (it exists — grep it), add a report-only gate
   `relative_error_context` for regression: MAE of the winner ÷ median(|target|) from the
   profile (the profile has target stats; verify field names). Not pass/fail — its details
   string states the ratio plainly ("typical error is 32% of the median target value").
5. Critic prompt: receives the three fields + the relative-error gate, with instructions: when
   tolerance is stated and violated → verdict must not be approve; when tolerance is absent for
   a regression task with relative error > a generous constant (e.g. 25%, named constant with
   rationale) → findings MUST include a severity=high finding saying fitness is unknown and the
   business tolerance must be supplied. The verdict may still be approve (beats-dummy remains a
   valid floor) but the finding forces the gap into the report and Gate 3 dossier.

**Tests:** ProblemSpec round-trips new fields; requirement extraction with a goal text that
states tolerance ("within 10% for pricing") populates it and one without leaves it empty (use
`FixtureTransport.from_texts` with canned responses — assert the SPEC fields, not the LLM);
`relative_error_context` computes the documented ratio; critic fixture-driven test: with a
stated-and-violated tolerance the fixture critique's approve is overridden/flagged per the
implementation choice (decide: deterministic post-check on the critique, NOT trust in the
prompt — a `_fitness_findings()` that appends the high-severity finding deterministically when
the ratio test fails; prompts steer, code enforces).

Commit: `feat(critic): capture decision context and judge fit-for-purpose, not just beats-dummy`

---

## Task 4 — Structured, evidence-cited iteration actions

**Problem.** Iteration = model-space search. The critique's `iteration_actions` are free-text
prose; the executor can only produce new ExperimentSpecs from them. The system cannot decide
"transform the target" or "stop, more experiments won't help," and nothing forces its proposals
to be grounded in evidence.

**Change.**
1. Extend the plan-revision wire format (grep `revise_experiment_plan` in
   `src/mlteam/agents/`): revised specs may carry `target_transform` (Task 2's field — the
   validator whitelists it). This alone gives iteration its first problem-space move.
2. The revision prompt receives, alongside prior results: the EvaluationReport's
   `slice_metrics`/leakage details and the fitness findings from Task 3 — the evidence that
   exists at iteration time (do NOT add new OOF computation for regression here; that cost was
   deliberately removed on 2026-07-08).
3. Grounding rule, enforced in code not prompt: each revised spec must include a
   `rationale` citing which prior experiment_ids or finding categories motivated it (extend the
   wire schema); the validator rejects (drops loudly, existing convention) specs whose cited
   experiment_ids don't exist in this run's results. Prompts steer; validation enforces.
4. Explicit stop: the revision response schema gains `stop: bool` + `stop_reason` — when the
   model judges further experiments not worthwhile, the loop records the reason in the
   IterationRecord and breaks (today only budget/verdict can stop it).

**Tests** (`tests/test_iteration.py` extensions): a fixture revision citing a nonexistent
experiment id → spec dropped with warning, run continues; a fixture revision with
`stop: true` → loop ends, reason recorded; a fixture revision setting `target_transform` on a
new spec → the retrained result carries it (ties Task 2's plumbing to the loop).

Commit: `feat(iteration): evidence-cited revisions, problem-space moves, and an explicit stop verb`

---

## Task 5 — High-cardinality categoricals via cross-fitted target encoding

**Problem.** Feature engineering is limited to row-local stateless ops (the correct leakage
guardrail), which walls off the highest-leverage tabular technique. One stateful transform is
available leakage-safe TODAY without building the full effect system:
`sklearn.preprocessing.TargetEncoder` cross-fits internally during `fit_transform`, and running
INSIDE the pipeline means outer CV/holdout discipline already applies.

**Change.** In `src/mlteam/templates/registry.py`'s preprocessor builders (grep
`build_preprocessor` / `build_scaled_preprocessor`): categorical columns with cardinality above
a named constant (e.g. `_TARGET_ENCODE_MIN_CARDINALITY = 20`, rationale: below this one-hot is
dense enough and more interpretable) route to `TargetEncoder` instead of one-hot/ordinal.
Cardinality is computable from the roles/frame at build time — check how `_column_roles` feeds
builders and pick the least invasive point to count uniques. Applies to both classification and
regression (sklearn's encoder handles both; verify the sklearn version in uv.lock supports
`target_type="auto"`). CatBoost templates are exempt (native categorical handling already —
see `CatBoostCategoricalWrapper`).

**Leakage proof, not leakage hope** (required test): on data where the categorical is pure noise
(target shuffled), the encoded model's CV primary metric must NOT beat the dummy floor beyond
noise — cross-fitting is what prevents the classic target-encoding leak, and this test pins it.
Plus: flip the `high_cardinality_categorical` trap from xfail.

Commit: `feat(templates): cross-fitted target encoding for high-cardinality categoricals`

---

## Task 6 — Value-of-computation v0: effort scales with verdict

**Problem.** After a halt/bad verdict, the platform spends the same ~hour on rigorous
explainability as it would for a shippable model ("polishing a turd", observed live).

**Change.** In `run_manager._execute` (and demo.py): when the final verdict is HALT →
explainability runs in cheap mode; when ITERATE-exhausted-budget with the winner failing Task
3's relative-error context → also cheap mode. Cheap mode = new `explain_model(...,
effort="cheap")` parameter: skips SHAP and PDP figures entirely, permutation importance with
`n_repeats` reduced (named constants; document that importance std widens accordingly in the
figure title). Full mode remains the default and the approve-path behavior. Log the downgrade
decision via `reporter.log` with the reason — the user must see WHY the report is thinner.

**Tests:** halt-verdict path produces an ExplainabilityReport with importances but no
SHAP/PDP figures and the log line present; approve path unchanged (figure kinds as today —
existing phase5 assertions must still pass untouched).

Commit: `feat(api): explainability effort scales with verdict -- stop polishing unshippable models`

---

## Task 7 — Lessons substrate v0: retro → durable propositions → planner priors

**Problem.** The ledger stores winner-template priors only. "Log-transform price-like targets"
— the lesson of the founding incident — has nowhere to live, so every failure is re-litigated
from scratch. The platform's only learning loop today is a human asking an assistant to patch
code.

**Change.**
1. New `src/mlteam/store/lessons.py`, mirroring `ledger.py`'s shape exactly (append-only JSONL
   at `store_root / "lessons.jsonl"`, never-raises writes, typed entries):
   `{"lesson": str (<= 300 chars, imperative), "evidence": str, "task_type": str,
   "source_run_id": str, "timestamp": str}`.
2. New retro agent `src/mlteam/agents/retro.py`: one best-effort LLM call after a run finishes
   (both approve and halt — halts teach more), given the critique, outcomes summary, triage
   remedies, and fitness findings; returns 0–3 lessons (wire-validated: length caps, no
   free-form fields). Recorded only for live-transport runs (same policy as the ledger's
   fixture exclusion — grep `_record_ledger_outcome` for where transport-awareness lives after
   REPORT_INTEGRITY Task 7 lands; if it hasn't landed, gate on the same condition inline and
   note the dependency in the commit body).
3. Consumption: `plan_experiments` and the cleaning proposer include the top-K
   (K=5, constant) most recent lessons matching the task_type in their prompt payload, clearly
   framed as "lessons from previous runs on this store — weigh, don't blindly obey".
4. Retro failures never fail a run (narrow catch, logged) — same contract as the ledger.

**Tests:** lessons round-trip through the JSONL; retro with a fixture response records
validated lessons and rejects malformed ones loudly; planner payload includes matching-task
lessons and excludes other-task ones; a fixture-transport run records nothing.

Commit: `feat(agents): post-run retro distills lessons that feed future planning`

---

## Task 8 (stretch) — Stage-level override: bring-your-own split/features

A teammate accepts partial delegation; an all-or-nothing pipeline doesn't. Smallest useful
increment: `run_tabular_pipeline`/API launch accept an optional user-supplied
`holdout_indices` file (or a `split_column` naming a column with train/holdout labels), which
the split strategist honors verbatim (recording `strategy="user_supplied"` and skipping its own
logic, with the holdout guard applying identically). Defer feature injection. Only attempt
after Tasks 0–7; the design above is a sketch, not a decision — validate against
`plan_split`'s actual interface first.

---

## Deferred by design (do not attempt as side effects)

- Full effect-type system for arbitrary stateful features (Task 5 deliberately rides sklearn's
  cross-fitting instead).
- Forecasting target transforms (raw-scale lag features vs transformed fitting — needs real
  design).
- Adversarial validation / temporal-risk detection (trap stays xfail with no task number).
- Calibration-tracked autonomy, distributed execution, GPU scheduling, non-tabular modalities.
- Anything in `REPORT_INTEGRITY_ROADMAP.md` tasks 5–7 and 9–12 — separate track, separate
  commits.
