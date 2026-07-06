# Scientific & Engineering Audit — Findings and Implementation Roadmap

**Auditor stance:** principal ML architect / statistician / Q1-journal reviewer attempting to
reject the system. Every claim below was verified against the actual source (file:line),
math re-derived from first principles, and where possible confirmed against the LIVE running
API. No component was trusted because it exists or because its tests pass.

**Audience:** an AI coding agent (Sonnet) executing the roadmap at the bottom, one task per
commit. **This document contains findings and specifications only — no code.**

**Date:** 2026-07-07. Audited tree: `main` @ `e88a070`.

---

## Part 0 — What was independently verified and found CORRECT

A rejection-minded review must also state what stands, so the implementer does not "fix"
sound components:

| Component | Verdict |
|---|---|
| Preprocessing inside sklearn `Pipeline`, re-fit per CV fold (`templates/registry.py`) | Correct — no fit-outside-fold leakage in any template |
| Cleaner restricted to stateless ops (drop_column / dedup / missing-token normalize), learned transforms banned by whitelist (`agents/cleaner.py:44`) | Correct and unusually disciplined |
| Split methodology: StratifiedKFold for classification, KFold for regression, TimeSeriesSplit for forecasting; temporal holdout cut on a **timestamp boundary** not a row count (`agents/splitter.py:126-141`) | Correct; the duplicate-timestamp boundary handling is better than most published pipelines |
| Pre-split dedup in triage (with the correct argument that post-split dedup cannot fix contamination) (`agents/triage.py`) | Correct |
| Lag/rolling features all `shift(>=1)`, warmup rows dropped, panel data fail-loud (`agents/time_features.py`) | Correct for one-step-ahead (see M3 for the horizon caveat) |
| Holdout guard: store-level denial of all reads until `final_reporting_access`; `holdout_untouched` computed from the guard's access log, never caller-asserted (`evaluation/report.py:71-75`) | Correct — this is a genuinely strong design |
| Determinism: LightGBM `deterministic+force_row_wise+n_jobs=1`, CatBoost `thread_count=1`, XGB `n_jobs=1`, seeded TPE, `n_jobs=1` CV, fold-assignment hash recorded in SplitPlan, env hash from `uv.lock` | Correct reproducibility story |
| Stacking uses internal `cv=3` OOF meta-features (no stacking leakage) | Correct |
| Critic hard gates enforced in code, LLM cannot approve past them (`agents/critic.py:_enforce_verdict`) | Correct mechanism (but see C1: one gate's *input* is wrong) |
| Dummy-floor mandatory, plan validator enforces baselines-first | Correct |
| RMSE/MAE/R²/AUC etc. all delegated to sklearn scorers with sign normalization (`evaluation/metrics.py`) | Correct — no hand-rolled metric math to get wrong |

---

## Part 1 — Findings

Severity ∈ {Critical, High, Medium, Low}; Priority ∈ {P0..P3}; Complexity ∈ {S, M, L}.

### C1 — Cross-run artifact attribution silently corrupts the critic's evidence (LIVE, observed)

- **Severity/Category/Priority:** Critical / Architecture + Statistics / **P0** (S)
- **Location:** `evaluation/results.py:63-72` (`list_experiment_outcomes`), plus the same
  first-writer filter in `api/views.py` (`experiments_view`, `critique_view`,
  `problem_spec_view`, `explainability_view`, `_evaluation_for_model`) and, transitively,
  `orchestration/pipeline.py:_completed_outcomes_by_key` (reuse lookup).
- **Evidence (reproduced live during this audit):** the currently-running API's demo run
  (`run-8c7001f186ab`, healthy synthetic churn, ROC-AUC 0.77 vs dummy 0.50) ended
  `verdict: halt` with the finding
  `critical | gate_override | missing outcomes for: ['screen-linear', 'exp-dummy']`.
  Those two experiments **ran and completed** — but their outcome artifacts are
  content-addressed (id excludes `run_id`), byte-identical to an earlier seeding's, so
  `INSERT OR IGNORE` left them attributed to the older run. `list_experiment_outcomes`
  filters `record.run_id == run_id` → the new run sees holes → the `complete_experiment_field`
  HARD gate force-halts a perfectly good run.
- **Impact:** healthy runs are halted; leaderboards/problem-spec/critique/explainability
  views go empty for reruns over identical data; the reuse lookup finds nothing and
  retrains; the deployment dossier under-reports the field. This is the root cause of the
  intermittent `critic_verdict_overridden enforced=halt` seen in prior testing. The fix
  infrastructure **already exists**: the `run_artifacts` membership table added in
  DASHBOARD_UX_PLAN Phase 4 — but only `lineage_graph`/`list_artifacts` use it.
- **Correction:** one shared run-scoping helper (membership table with legacy
  `record.run_id` fallback), used by **every** run-scoped artifact query. See Task 1.

### C2 — Regression leak detector compares unnormalized MI (nats) to a 0.9 threshold

- **Severity/Category/Priority:** Critical (false HALTs) / Statistics / **P1** (S)
- **Location:** `agents/profiler.py:131-138`; consumed by `evaluation/leakage.py:21-43`
  and the `SUSPECTED_LEAK` critical warning.
- **Why it is wrong (first principles):** for classification the code correctly normalizes
  `MI / H(y)` and clips to [0,1] — for a binary target `H(y) ≤ ln 2 ≈ 0.693` nats, so the
  normalization is what makes the 0.9 threshold meaningful. For **regression** the raw
  `mutual_info_regression` output is returned unnormalized. Differential-entropy-based MI
  for a continuous target is unbounded: a strong-but-legitimate predictor (|r| ≈ 0.9 ⇒
  Gaussian MI = −½ln(1−r²) ≈ 0.83 nats; |r| ≈ 0.95 ⇒ 1.16 nats) crosses 0.9 with **no
  leakage whatsoever**. The same constant means two different things in the two branches.
- **Impact:** legitimate strong-signal regression datasets trip a CRITICAL
  `SUSPECTED_LEAK` warning → `near_perfect_features` hard gate → forced HALT. Conversely
  the threshold is not calibrated to catch anything specific.
- **Correction:** convert regression MI to the Gaussian-equivalent correlation
  `r_MI = sqrt(1 − exp(−2·MI))` (Joe, 1989) and flag at `r_MI > 0.99` — consistent with
  the existing Pearson 0.99 threshold. See Task 3.

### H1 — "Significant" is an ad hoc heuristic; two conflicting "pooled std" definitions; population variance on 3 folds

- **Severity/Category/Priority:** High / Statistics / **P1** (M)
- **Locations:**
  - `evaluation/metrics.py:73-82` — `summarize_folds` divides by `n` (population variance).
    With k=3 folds this understates fold σ by a factor √(2/3) ≈ 0.82. For inference from a
    sample of folds, `ddof=1` is required.
  - `evaluation/report.py:24,39` — significance := `improvement > 2.0 · sqrt(s_m² + s_d²)`.
  - `evaluation/field_checks.py:101` — margin-in-stds uses `sqrt((s1² + s2²)/2)`.
    The two modules disagree by a factor of √2 on what one "std" is, so the critic's
    "0.5 stds = tie" guidance and the report's "significant" gate are not on the same scale.
- **Why it is wrong (first principles):** both models were cross-validated on **identical
  folds** (same seed → same `StratifiedKFold` assignment; `MetricSummary.fold_values` is
  stored per fold, in order). The correct object is the **per-fold paired difference**
  d_i = m_i − b_i, whose variance is typically far smaller than the pooled marginal
  variance (positive correlation between models on the same fold). The independent-samples
  pooling both mis-states the scale and ignores pairing. Additionally, CV fold scores are
  not independent (overlapping training sets), so even a paired t-test is anti-conservative
  without correction — the standard remedy is the Nadeau–Bengio corrected resampled t-test:
  `t = d̄ / sqrt( s_d² · (1/k + n_test/n_train) )` with k−1 df (Nadeau & Bengio 2003;
  Bouckaert & Frank 2004). Presenting the current 2σ rule as "significantly outperforms" in
  a business-facing report (`reporting/builder.py:_headline_sentence`) is the kind of claim
  a reviewer strikes.
- **Impact:** the `beats_dummy_significantly` HARD gate, the tie-detection the critic is
  instructed to reason from, and the report's headline all rest on a scale that is (a)
  internally inconsistent, (b) biased small, (c) not a test.
- **Correction:** paired fold deltas + Nadeau–Bengio corrected t and p-value; a single
  shared effect-size definition; honest wording fallback when pairing is unavailable.
  See Task 2.

### H2 — Holdout evaluation: cleaning manifest not replayed; point estimates with no uncertainty

- **Severity/Category/Priority:** High / ML correctness + Statistics / **P1** (M)
- **Location:** `reporting/builder.py:59-89` (`_holdout_metrics`).
- **Evidence (a):** the feature plan IS replayed on the holdout (Phase-6 fix), but the
  cleaning manifest is NOT — despite `execute_cleaning`'s own docstring promising "the
  manifest is replayable on any schema-compatible dataset (e.g. the holdout at
  final-reporting time)" (`agents/cleaner.py:221-225`). Concretely, in the demo dataset
  `contract_type` contains textual `"N/A"`: training data had it normalized to real nulls
  (imputer handles them); holdout rows still carry the literal string `"N/A"`, which the
  encoder treats as an unseen category. Training-serving skew inside our own evaluation:
  holdout metrics are computed under a **different preprocessing** than training.
- **Evidence (b):** holdout metrics are single point estimates printed to 3 decimals. The
  demo holdout is ~48 rows; the standard error of an AUC at n≈48 is roughly 0.06-0.08 —
  the third decimal is noise, and the report offers no interval.
- **Correction:** replay the stateless cleaning ops (missing-token normalization,
  drop_column; NOT dedup — the evaluation population should keep its natural duplicates)
  before the feature plan; add seeded bootstrap percentile CIs (B=1000) per holdout metric;
  print holdout n. See Task 4.

### H3 — HPO winner's curse: search-optimized CV score reused as the comparison score

- **Severity/Category/Priority:** High / Statistics / **P1** (S)
- **Location:** `agents/hpo.py:141-237`, `orchestration/iteration.py:158-187`.
- **Why it is wrong:** Optuna maximizes mean CV score across trials; the maximum of many
  noisy estimates is upward-biased (Cawley & Talbot 2010). The winning config is then
  retrained by `train_experiment` **on the same folds the search optimized** (same
  seed+iteration), and that biased score competes against the defaults' unbiased scores in
  `_best_key`. Additionally, HPO runs on `seed + iteration` folds while incumbents used
  `seed` folds, so the final comparison mixes scores from different fold partitions —
  breaking the pairing needed for H1's statistics.
- **Correction:** search on `seed + iteration` (fine — decorrelates the search), but
  **re-score the best config with the base seed's folds** for the reported metrics and
  selection. One extra CV of a single config; restores pairing; removes most selection
  bias. See Task 5.

### H4 — Permutation importance: wrong data, wrong scorer; residual analysis on train residuals

- **Severity/Category/Priority:** High / Explainability / **P2** (M)
- **Location:** `agents/explainer.py:44-64` (and `_residual_analysis`, 112-187).
- **Why it is wrong:** (a) importance is computed on the **training frame with the final
  model fitted on that same frame** — for overfit-capable models this rewards memorized
  features and is explicitly warned against in the sklearn documentation; (b)
  `permutation_importance(..., scoring=None)` defaults to accuracy/R², NOT the run's
  primary metric — for imbalanced churn selected on ROC-AUC/PR-AUC, importances answer a
  different question than the one the report claims; (c) train residuals of boosted trees
  are shrunk toward zero, making "worst slice by train MAE" close to meaningless.
- **Correction:** out-of-fold importances (fit per fold, permute on the fold's test rows,
  average) with `scoring=` the primary metric; OOF residuals for slices. See Task 6.

### H5 — SHAP bars can be mislabeled: original column names paired with transformed feature matrix

- **Severity/Category/Priority:** High / Explainability / **P2** (S)
- **Location:** `agents/explainer.py:91-99`.
- **Why it is wrong:** `transformed = preprocess.transform(sample)` changes width whenever
  one-hot encoding is involved (`build_scaled_preprocessor`), but
  `feature_names = list(sample.columns)` keeps the original columns. The bar chart pairs
  element-wise → silently truncated/misattributed importances. Also `values[-1]` assumes
  the list API; newer SHAP returns a 3-D array where `[-1]` selects the last **sample**.
- **Correction:** `preprocess.get_feature_names_out()`; handle both list and 3-D array
  returns explicitly. See Task 6.

### H6 — Upload filename path traversal (security)

- **Severity/Category/Priority:** High / Engineering-Security / **P2** (S)
- **Location:** `api/app.py:159-161` — `settings.upload_dir / f"{ui_run_id}-{file.filename}"`.
- **Why it is wrong:** `file.filename` is client-controlled; a name containing path
  separators (`../../evil.csv`) escapes `upload_dir` on write. Local-tool posture reduces
  exposure, but the fix costs three lines.
- **Correction:** `Path(file.filename).name`, allowlist suffixes (.csv/.parquet), cap
  length; also cap the `PATCH` run-name length. See Task 10.

### H7 — Cost asymmetry is solicited at Gate 1, then ignored everywhere

- **Severity/Category/Priority:** High / ML methodology / **P2** (L, shared with H8)
- **Evidence:** `grep cost_asymmetry src/` → captured in `requirement.py`, stored in
  `ProblemSpec`, displayed at Gate 1 — zero reads in training, threshold selection,
  metric weighting, evaluation, or reporting. A human confirms FP:FN = 1:5 and the system
  proceeds to optimize threshold-free rank metrics and report accuracy/F1 at an implicit
  0.5 cutoff. Soliciting a business input and silently discarding it is worse than not
  asking: it manufactures false confidence that the model is cost-aware.
- **Correction:** compute the cost-optimal decision threshold from out-of-fold
  probabilities (minimize `FP·c_fp + FN·c_fn`), store it, report the confusion/costs at
  that threshold on the holdout. See Task 7.

### H8 — Calibration gate theater: the ECE gate can never fire

- **Severity/Category/Priority:** High / Statistics + Governance / **P2** (shared with H7)
- **Evidence:** `EvaluationReport.calibration` (`contracts/evaluation.py:46-50,78`) is
  never written by any producer; `critic.py:172-180` gates on
  `ece is None or ece <= 0.10` → always "not measured" → always passes; the critic's
  prompt even instructs it to weigh calibration. A gate that structurally cannot fail is
  audit theater and would be flagged by any reviewer.
- **Correction:** compute Brier + binned ECE from the same OOF probabilities Task 7 needs
  (one shared OOF pass). If not computed for a task type, REMOVE the gate for that run
  rather than emitting a vacuous pass. See Task 7.

### M1 — Winner selection ignores variance and multiplicity; dossier margin can describe a different model than the one deployed

- Medium / Statistics / P2 (S). `_select_best` takes the max of CV means over the field
  (winner's curse across 5-8+ candidates grows with iterations); nothing in the report
  states how many candidates the winner was selected among. Separately,
  `deployer.build_deployment_dossier` computes `winner_margin` from `ranked[0]` over ALL
  completed outcomes (screening included) while `is_winner` marks `model.experiment_id` —
  when they differ, Gate 3's margin table describes a model that is not the one being
  signed off. Correction: compute the margin against `model.experiment_id`, and disclose
  "selected among N candidates" in dossier + report. See Task 8.

### M2 — Screening results are shown to everyone but excluded from selection

- Medium / ML methodology / P2 (M). `_run_screening`'s trained results are discarded from
  the candidate pool (`iteration.py` seeds `results` from planned specs only). On the
  recorded demo data, `screen-linear` scores ROC-AUC 0.772 vs the crowned
  `exp-lgbm-default` 0.769 — the field's best model is not selectable, while still
  appearing in the outcome table the critic reads (and skewing `compute_winner_margin`
  negative). Either include screening outcomes in the pool (recommended — they are
  first-class TrainedModels) or exclude them from margin candidates; do not keep the
  inconsistent middle. **Warning to implementer:** including them changes the demo-fixture
  winner (linear beats lgbm) — `test_agentic_pipeline.py`'s
  `best_template_id == "lightgbm_default"` assertion and demo expectations must be
  updated deliberately, not papered over. See Task 8.

### M3 — `prediction_horizon` exists in the spec but the forecasting stack is silently one-step-ahead

- Medium / ML correctness / P2 (M). `time_features.py` builds `lag_1..` features
  unconditionally; a user who declared a 7-step horizon gets metrics that assume the
  previous step's true target is known at prediction time — an unrealizable forecast.
  Correction: when `h > 1`, start lags at `shift(h)` and shift rolling windows by `h`;
  always print the horizon assumption in the report. See Task 9.

### M4 — API path hardcodes weak evidence settings

- Medium / MLOps / P2 (S). `run_manager.py` fixes `n_folds=3`, `hpo_trials=8`,
  `RunBudget(max_iterations=2)` outside configuration. k=3 population-variance folds is
  the weakest defensible CV; combined with H1 it makes every "significant" claim fragile.
  Move to `APISettings` (env-overridable), default `n_folds=5`. See Task 10.

### M5 — Pearson-only leak correlation misses monotone leaks

- Medium / Statistics / P2 (S, rides with Task 3). A leaked probability or rank transform
  of the target evades `|pearson| > 0.99` while being a perfect monotone predictor. Add
  Spearman alongside Pearson in `_target_relationships` and gate on either. See Task 3.

### M6 — Imbalance warning never changes behavior

- Medium / ML methodology / P2 (S, rides with Task 7). Profiler emits `TARGET_IMBALANCE`;
  no default class weighting, no threshold move, and `ModelConfigDraft`'s whitelist does
  not expose `class_weight`, so even the planner cannot act on its own warning. Whitelist
  `class_weight` (sklearn/LGBM-compatible values) + prompt nudge. See Task 7.

### M7 — SSE stream holds one threadpool thread per subscriber

- Medium→Low / Engineering / P3. `_sse` is a sync generator (Starlette runs it in the
  default ~40-thread pool) with a 15 s blocking `q.get`. Dozens of open dashboard tabs
  starve the API. Acceptable single-user; convert to an async generator with
  `asyncio.Queue` if/when multi-user. Backlog note in Task 11.

### M8 — Drift monitoring exists as a function but is not reachable

- Medium / MLOps-Deployment / P3. `evidently_drift_report` (`deployer.py:301`) has no API
  route, no serving-app hook, no dashboard surface — the deployment story claims drift
  readiness the product cannot exercise. Backlog: `POST /api/runs/{id}/drift` accepting a
  CSV. Task 11.

### Low (grouped)

- **L1** `QuantileGBMRegressor` does not enforce quantile non-crossing (sort at predict).
- **L2** CatBoost is fed ordinal codes instead of its native categorical handling — the
  one thing CatBoost is for; pass cat feature indices where feasible.
- **L3** Ledger `matching_priors` does not filter by `primary_metric`, so a planner prior
  can mix rmse-margin entries with auc-margin entries (entries do carry the metric name —
  the payload should group by it).
- **L4** `experiments_view.is_baseline` derives from `template == "dummy_baseline"`,
  contradicting the plan's `is_baseline=True` on `lightgbm_default`.
- **L5** Imbalance denominator uses full height incl. null-target rows (`profiler.py:234`).
- **L6** `_headline`'s mixed holdout/CV cards are labeled, but the business template
  should visually separate "out-of-sample" from "CV" claims.
- **L7** FastAPI `on_event` deprecation; migrate to lifespan handlers opportunistically.
- **L8** Tie-break in `_select_best` is first-in-spec-order — deterministic but
  undocumented; add one comment line.

---

## Part 2 — Implementation roadmap (hand to Sonnet, execute in order)

**Ground rules (same discipline as prior plans):** every task ends with
`uv run ruff check src tests && uv run pytest -q` green from the repo root and
`npx tsc --noEmit` clean in `dashboard/` (only Tasks 8/10/11 touch the dashboard);
ONE conventional commit per task; push after each; restart the API after backend changes.
The recorded LLM fixtures replay by **call order**, not payload content — payload changes
are safe, adding/removing/reordering LLM calls is not. None of these tasks adds an LLM call.

### Task 1 — Run-scoped artifact queries everywhere (fixes C1) — P0, S

**Objective:** every "artifacts of run X" query uses the `run_artifacts` membership table
(with legacy `artifacts.run_id` fallback), not first-writer attribution.
**Modules:** `store/guard.py` or `store/interface.py` (new shared helper, e.g.
`run_artifact_ids(store, run_id) -> frozenset[str] | None` mirroring
`api/views.py:_run_scoped_ids`), `evaluation/results.py:list_experiment_outcomes`,
`api/views.py` (`experiments_view`, `_legacy_experiments_view`, `_evaluation_for_model`,
`critique_view`, `problem_spec_view`, `explainability_view` — replace every
`record.run_id == run_id` filter), keep `orchestration/pipeline.py` unchanged (it inherits
the fix through `list_experiment_outcomes`).
**Steps:** (1) hoist the scoping helper next to the store so views and evaluation share
one implementation; (2) swap all seven call sites; (3) regression tests: in ONE store, run
two fixture pipelines over identical data/seed; assert the second run's
`list_experiment_outcomes` contains ALL its experiment ids (incl. screening), its
`experiments_view` is fully populated, and `critique_evaluation`'s terminal-state audit
reports complete; (4) live verification: delete nothing, just re-seed the demo against the
existing persistent `artifact_store` — the critic must **not** halt with
`missing outcomes`.
**Acceptance:** the exact live failure reproduced in this audit (demo halt,
`missing outcomes for: ['screen-linear', 'exp-dummy']`) no longer occurs on a warm store.
**Commit:** `fix: scope all run artifact queries via membership table (kills false critic halts)`

### Task 2 — Real paired statistics; one margin definition (fixes H1) — P1, M

**Objective:** replace the 2σ heuristic with a paired Nadeau–Bengio corrected t-test and
make every "in stds" number mean the same thing.
**Modules:** `evaluation/metrics.py` (`summarize_folds` → `ddof=1`, guard k<2), new
`evaluation/stats.py` (pure functions: `paired_fold_deltas`, `nb_corrected_t`,
`p_value` from t with k−1 df — `scipy.stats.t.sf`, scipy is already a transitive dep;
correction multiplier `(1/k + n_test/n_train)` with `n_test/n_train = 1/(k−1)` for k-fold),
`evaluation/report.py` (`BaselineDelta`: add optional `p_value`, set
`significant := direction-correct AND p < 0.05` when fold pairing is available —
identical fold count and same fold seed — else fall back to the current heuristic and say
so in `details`), `evaluation/field_checks.py` (keep the effect-size framing but compute
it ONE way — `sqrt((s1²+s2²)/2)` with ddof=1 inputs — and rename the field docstrings to
"pooled-σ effect size"; add paired `d̄`/`s_d` when fold values align),
`agents/critic.py` SYSTEM_PROMPT tie wording, `reporting/builder.py:_headline_sentence`
("significantly outperforms" only when the test fired; otherwise "outperforms").
**Contracts note:** `BaselineDelta` gains an optional field — additive, no migration.
**Tests:** hand-computed cases (e.g. deltas (0.02,0.03,0.01), k=3 → t and p verified
manually in the test docstring); a near-tie that the old heuristic called significant and
the corrected test does not; ddof change ripples into existing hand-computed margin tests
(`test_critic_field_evidence.py` uses symmetric two-fold values — recompute expectations
deliberately).
**Commit:** `feat: paired Nadeau-Bengio corrected significance + consistent effect sizes`

### Task 3 — Normalize regression MI; add Spearman (fixes C2, M5) — P1, S

**Modules:** `agents/profiler.py` (`_target_relationships`: regression branch →
`r_mi = sqrt(1 − exp(−2·mi))`, store that in `mutual_information` so it is bounded [0,1);
compute Spearman via rank-transform + corrcoef alongside Pearson; flag when either
`r_mi > 0.99` or `|spearman| > 0.99` or `|pearson| > 0.99`), `evaluation/leakage.py`
(threshold text), `contracts` only if a `spearman` field is added to
`TargetRelationship` (additive).
**Tests:** regression target with a legit r≈0.9 predictor → NO leak flag; `y + ε` clone
(r≈0.999) → flagged; monotone-but-nonlinear leak (`exp(y)`) → caught by Spearman, missed
by Pearson (assert both, documenting the reason for Spearman).
**Commit:** `fix: bounded MI leak scores for regression + Spearman monotone leak check`

### Task 4 — Holdout scoring parity + bootstrap CIs (fixes H2) — P1, M

**Modules:** `reporting/builder.py` (`ReportInputs` gains
`cleaning_manifest: CleaningManifest | None`; `_holdout_metrics` replays, in manifest
order, ONLY `standardize_missing_tokens` and `drop_column` ops — explicitly skip
`drop_duplicate_rows` with a comment on why the eval population keeps duplicates — then
the feature plan, then scores; add `_bootstrap_ci(pipeline, df, target, scorer, B=1000,
seed)` percentile CIs; each metric row gains `ci_low`/`ci_high`/`n`),
`reporting/templates.py` (render CI + holdout n), callers pass the manifest
(`api/run_manager.py`, `api/demo.py` — `store.get(pipeline.cleaning_manifest_id)`).
**Tests:** a dataset with textual "N/A" in a categorical → holdout metrics equal those on
a manually-normalized holdout frame (this is the regression that the current code fails);
CI fields present, `ci_low ≤ value ≤ ci_high`; bootstrap seeded/deterministic.
**Commit:** `fix: replay cleaning manifest on holdout + bootstrap CIs for holdout metrics`

### Task 5 — Honest HPO re-scoring (fixes H3) — P1, S

**Modules:** `agents/hpo.py` (`tune_experiment` gains `report_seed: int | None = None`;
the study keeps `seed`, the final `train_experiment` uses `report_seed or seed`),
`orchestration/iteration.py` (pass `seed=seed + iteration` for search,
`report_seed=seed` for the retrain — restoring fold pairing with incumbents).
**Tests:** with a fixed dataset, assert the retrained model's fold count/seed matches the
base experiments' (pairing restored); assert reported best score comes from the re-scored
CV, not `study.best_value` (they may legitimately differ — the HPOOutcome should carry
both, `search_cv_score` vs `reported_cv_score`, additive dataclass fields).
**Commit:** `fix: re-score HPO winner on base folds (removes winner's-curse comparison)`

### Task 6 — Valid explainability: OOF importances, correct SHAP names, OOF residuals (fixes H4, H5) — P2, M-L

**Modules:** new `evaluation/oof.py` (shared helper: given store/dataset/template/config/
seed/n_folds, produce OOF predictions + per-fold fitted estimators via the same
`_make_cv`; reused by Task 7 — build it here), `agents/explainer.py`
(`explain_model` gains `primary_metric: str`; permutation importance per fold on that
fold's test rows with `scoring=get_scorer(scoring_for(task)[primary_metric])`, averaged
across folds; SHAP: `feature_names = preprocess.get_feature_names_out()`, handle list vs
3-D ndarray (`values[..., class_idx]` when `ndim == 3`); residual slices from OOF
predictions), callers (`api/run_manager.py`, `api/demo.py`) pass the primary metric.
**Tests:** one-hot template → SHAP name count equals transformed width; importance scorer
respected (a feature predictive for AUC but not accuracy ranks under AUC scoring);
OOF residual slice test on a synthetic segment with injected noise.
**Commit:** `fix: out-of-fold importances/residuals + correct SHAP feature naming`

### Task 7 — Cost-aware thresholds + real calibration (fixes H7, H8, M6) — P2, L

**Modules:** `evaluation/oof.py` (from Task 6) supplies OOF probabilities;
new `evaluation/calibration.py` (Brier, 10-bin ECE) and
`evaluation/thresholds.py` (cost-optimal threshold minimizing
`c_fp·FP(t) + c_fn·FN(t)` over the OOF ROC sweep); `evaluation/report.py` populates
`EvaluationReport.calibration` and a new additive `decision_threshold` block (threshold,
expected cost at threshold, costs used); `agents/critic.py` (calibration gate: when a
classification run has OOF proba, gate on the real ECE; when not computed, DROP the gate
from the list instead of emitting a vacuous pass); `reporting/builder.py` (holdout
confusion + total cost at the chosen threshold, alongside the rank metrics);
`agents/experiment_planner.py` (`ModelConfigDraft` whitelist gains `class_weight`;
prompt: on TARGET_IMBALANCE consider class weights); `reporting/templates.py` rendering.
**Fixture caution:** planner prompt text changes are safe; do not add LLM calls.
**Tests:** synthetic imbalanced set with known asymmetry (c_fn=5·c_fp) → threshold < 0.5
and expected cost at chosen threshold ≤ cost at 0.5; ECE hand-computed on a tiny
constructed proba set; critic gate list contains `calibration_acceptable` only when
measured.
**Commit:** `feat: cost-optimal decision threshold + measured calibration (ECE/Brier)`

### Task 8 — Selection integrity: screening in the pool, aligned dossier, disclosure (fixes M1, M2) — P2, M

**Modules:** `orchestration/agentic.py` (merge screening `TrainResult`s into
`pipeline.train_results` under their `screen-*` ids so `_select_best`/`_best_key` see
them), `agents/deployer.py` (dossier margin computed against `model.experiment_id`, not
`ranked[0]`; add `"selected_among": N`), `reporting` (one line: "winner selected among N
trained candidates").
**Deliberate consequence (do not paper over):** on the recorded churn fixture,
`screen-linear` (0.772) now beats `exp-lgbm-default` (0.769) — the demo winner changes to
the linear model. Update `test_agentic_pipeline.py` expectations and any demo assertions
knowingly; the honest winner IS the point of this task.
**Tests:** constructed outcomes where a screening probe wins → it is selected and the
dossier margin references the deployed (screening) model; disclosure count correct.
**Commit:** `fix: screening models compete for selection; dossier margin matches deployed model`

### Task 9 — Forecast horizon honesty (fixes M3) — P2, M

**Modules:** `agents/time_features.py` (`add_forecast_features(..., horizon: int = 1)`:
lags become `shift(h), shift(h+1), ...` from the vetted set ≥ h; rolling windows shift by
`h`), `orchestration/agentic.py`/`pipeline.py` (pass
`spec.prediction_horizon or 1`), `reporting/templates.py` ("metrics are h-step-ahead"
line), `agents/requirement.py` prompt already captures horizon — verify it lands in the
spec. Seasonal-naive template reads `lag_{season}` columns — ensure season ≥ h or the
template errors loudly (existing fail-loud path covers it).
**Tests:** h=7 → no `lag_1..lag_6` columns exist; h=1 unchanged byte-for-byte vs today;
report renders the horizon line.
**Commit:** `feat: honor prediction_horizon in lag construction and reporting`

### Task 10 — Security & configuration hygiene (fixes H6, M4) — P2, S

**Modules:** `api/app.py` (sanitize `Path(file.filename).name`, suffix allowlist
{.csv,.parquet}, filename length cap; `RenameRequest.name` max_length≈120),
`api/config.py` (`n_folds` default 5, `hpo_trials`, `max_iterations`,
`max_llm_cost_usd` as settings), `api/run_manager.py` reads them.
**Tests:** upload with `..%2F`-style names lands inside `upload_dir`; wrong suffix → 400;
settings override respected.
**Commit:** `fix: upload path sanitization + evidence settings via configuration`

### Task 11 — Backlog batch (Low items; optional, in any order) — P3, S each

L1 quantile sort-at-predict; L2 CatBoost native categoricals; L3 prior payload grouped by
metric; L4 `is_baseline` from plan flag; L5 imbalance denominator on labeled rows;
L6 template visual split of holdout vs CV; L7 lifespan migration; M7 async SSE;
M8 `POST /api/runs/{id}/drift` (CSV upload → `evidently_drift_report` → summary JSON +
dashboard card).
**Commit(s):** individually as `chore:`/`feat:` per item.

### Sequencing rationale

Task 1 first: it repairs the **evidence supply** every later statistical fix depends on
(and it is the only finding actively breaking production today). Tasks 2-5 make the
numbers the platform prints defensible (test, scale, holdout parity, HPO bias) before
Tasks 6-7 add new numbers on the shared OOF infrastructure (built once in Task 6, consumed
in Task 7). Task 8 changes selection behavior and test expectations, so it lands after the
statistics it will be judged by. Tasks 9-10 are independent but touch files Tasks 2-7
modify, so they follow to minimize rebase friction. Task 11 is polish.

### Definition of done (whole roadmap)

- A rerun over an identical dataset on a warm store approves normally (no false halts),
  with a fully populated leaderboard and lineage.
- Every "significant" in UI or report is backed by a named test with a p-value, or is
  reworded.
- Holdout numbers carry CIs and are computed under training-identical preprocessing.
- No gate can pass vacuously; no captured business input is silently unused.
- Full suite + lint + tsc green after every task; no new LLM calls anywhere.
