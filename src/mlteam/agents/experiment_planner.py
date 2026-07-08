"""Experiment Planner agent: which templates to train, with budget estimates.

The LLM selects templates + configs from the vetted registry; deterministic
code computes budget estimates and enforces the structural rules (cheap
baselines ALWAYS first — the ExperimentPlan validator refuses anything else).
The plan then awaits HUMAN GATE 2 with its cost estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

from mlteam.contracts import (
    BudgetEstimate,
    DataProfile,
    DecisionRecord,
    ExcludedTemplate,
    ExperimentPlan,
    ExperimentSpec,
    ProblemSpec,
    TaskType,
    new_artifact,
)
from mlteam.llm.client import LLMClient
from mlteam.llm.summaries import profile_summary
from mlteam.store.interface import ArtifactStore
from mlteam.store.ledger import matching_priors
from mlteam.store.lessons import matching_lessons
from mlteam.templates.registry import get_template, list_templates

_log = structlog.get_logger(__name__)
# JUDGMENT_ROADMAP.md Task 7: matches ledger.matching_priors' own default.
_LESSONS_TOP_K = 5

SYSTEM_PROMPT = """You are the Experiment Planner of an autonomous ML engineering \
team. Select which vetted pipeline templates to train and with what config.

Rules:
- template_id must come from the provided registry list. You never write model \
code — you only pick templates and fill configs.
- The plan MUST start with the cheap baselines, in this order: dummy_baseline \
first, then lightgbm_default — both marked is_baseline=true. They are the sanity \
floor every candidate must beat.
- After the baselines, add 2-4 candidate experiments chosen for THIS dataset. \
Guidance: xgboost_default and catboost_default are strong general candidates \
(catboost especially on categorical-heavy data); tabpfn_default excels under \
~10k rows; random_forest_default/extra_trees_default are robust to noise; \
linear_default when relationships look near-linear or interpretability matters; \
svm_default and knn_default only on small datasets (<5k rows); mlp_default on \
larger numeric-heavy data; naive_bayes_default is classification-only; \
stacking_ensemble_default pushes the accuracy ceiling when budget allows and \
stacking_deep_default (5 base families + passthrough) pushes it further at ~2x \
the cost; for anomaly_detection also consider the dedicated detectors \
(isolation_forest_default, one_class_svm_default, local_outlier_factor_default) \
alongside the classifiers.
- For forecasting: ALWAYS include seasonal_naive_default as a candidate — it is \
the honest floor (predict last season's value) and any forecaster that cannot \
beat it is worthless. Add lightgbm_quantile_default when the business needs \
prediction intervals, not just point forecasts.
- Prefer an EMPTY config — every template ships vetted defaults, and HPO tunes \
the winner later. Only set config keys when the profile clearly demands it, and \
only keys valid for that model's constructor.
- If the profile flags target_imbalance, set class_weight="balanced" on at least \
one candidate that supports it (linear_default, svm_default, random_forest_default, \
extra_trees_default, lightgbm_default only — any other template rejects this key). \
Do not set it as the ONLY response to imbalance; still schedule the usual \
candidate slate.
- If the profile flags target_transform_recommended (regression only — the target \
spans multiple orders of magnitude or is heavily right-skewed), set \
target_transform="log1p" on at least one non-baseline candidate so its performance \
can be compared against the raw-fit baselines. Never set it for a non-regression \
task or without that profile flag present.
- rationale should reference the profile (size, imbalance, cardinality).
- excluded_templates: list the major template families you did NOT schedule and \
WHY (e.g. "svm_default: skipped, >20k rows makes its quadratic cost prohibitive"; \
"mlp_default: skipped, dataset is small and heavily categorical, trees will \
dominate"). Cover at least: a linear model, svm/knn, mlp, and any ensemble/\
stacking template not chosen — a reviewer must be able to see the plan's negative \
space, not just what was picked.
- prior_outcomes, when present, are past runs on similarly-sized data of the \
same task type: which template won and by how much it beat the runner-up. \
Priors may REORDER your candidates and adjust budgets (e.g. try a repeatedly- \
winning template earlier) — they may NEVER cause you to skip the mandatory \
baselines, drop a candidate family without recording it in excluded_templates, \
or treat a past winner as guaranteed; every planned experiment is still \
evaluated on ITS OWN merits against this run's data.
- screening_results are dummy_baseline/linear_default/lightgbm_default ALREADY \
trained on THIS dataset — real evidence, not a prior. Read them as diagnostic \
instruments, not competitors:
  * linear_default close to lightgbm_default → signal is mostly linear: prefer a \
  regularized linear model plus one strong GBDT; a deep stack is unlikely to earn \
  its cost here.
  * linear_default near dummy_baseline while lightgbm_default is clearly better → \
  nonlinear/interaction-heavy signal: prioritize the GBDT family and \
  stacking_ensemble_default/stacking_deep_default; mlp_default only if rows are \
  numerous and the data is dense-numeric (not heavily categorical).
  * linear_default OR lightgbm_default scores near-perfect (e.g. roc_auc/accuracy \
  > 0.995, or r2 > 0.995) → treat this as a LEAKAGE ALARM, not a win: say so \
  explicitly in rationale and keep the plan modest — do not schedule a large, \
  expensive candidate slate chasing a score that is probably too good to be true \
  (the Critic's hard gates will also check this, but flag it here first).
  * skip svm_default/knn_default above ~20k rows (their cost or quality degrades \
  quadratically); skip mlp_default under ~10k rows or when the profile is heavily \
  categorical — trees dominate there.
  * dummy_baseline and lightgbm_default specs you plan with EMPTY config reuse \
  the screening run outright (zero extra training cost) — this is a reason to \
  prefer the vetted default config for those two slots unless the profile clearly \
  demands otherwise.
- lessons_from_previous_runs, when present, are distilled propositions from past \
runs on this store with the SAME task_type (e.g. "log-transform price-like \
targets spanning multiple orders of magnitude") — weigh them, don't blindly \
obey: they are not this run's evidence, and every experiment you plan still \
needs its own justification grounded in THIS profile/screening_results, not \
just a cited lesson. A lesson may inform target_transform or config choices; \
it never overrides the baselines-first/coverage rules above.
Return only JSON matching the schema."""

# Rough wall-time heuristics (seconds) per 1k rows, by template.
_SECONDS_PER_1K_ROWS: dict[str, float] = {
    "dummy_baseline": 0.5,
    "lightgbm_default": 6.0,
    "tabpfn_default": 45.0,
    "xgboost_default": 8.0,
    "catboost_default": 15.0,
    "random_forest_default": 10.0,
    "extra_trees_default": 8.0,
    "adaboost_default": 8.0,
    "linear_default": 2.0,
    "svm_default": 30.0,  # quadratic in rows — heuristic assumes small data
    "knn_default": 3.0,
    "naive_bayes_default": 1.0,
    "mlp_default": 20.0,
    "stacking_ensemble_default": 45.0,  # 3 base models x internal cv=3
    "stacking_deep_default": 90.0,  # 5 base models x internal cv=3 + passthrough
    "isolation_forest_default": 4.0,
    "one_class_svm_default": 20.0,
    "local_outlier_factor_default": 5.0,
    "seasonal_naive_default": 0.5,
    "lightgbm_quantile_default": 18.0,  # three quantile models
}
_MIN_BUDGET_SECONDS = 2.0

# Templates whose underlying estimator constructor actually accepts
# class_weight (checked against sklearn/LightGBM's own __init__ signatures).
# AdaBoost, XGBoost, CatBoost, KNN, GaussianNB, and MLP have no such
# parameter and would raise a TypeError at fit time if it were passed.
_CLASS_WEIGHT_CAPABLE_TEMPLATES = frozenset(
    {
        "linear_default",
        "svm_default",
        "random_forest_default",
        "extra_trees_default",
        "lightgbm_default",
    }
)


class ModelConfigDraft(BaseModel):
    """Whitelisted hyperparameters — the LLM tunes within these knobs only.

    (The structured-output grammar cannot express open dicts; a typed subset
    is also exactly what 'agents decide, libraries execute' wants.) Knobs span
    the template families: boosting (n_estimators/learning_rate/...), CatBoost
    (iterations/depth/l2_leaf_reg), forests (min_samples_leaf/max_features),
    linear/SVM (C/alpha), KNN (n_neighbors). Unused knobs stay None.

    ``class_weight`` is the ONE concrete lever the planner has for the
    TARGET_IMBALANCE warning the profiler already emits (audit finding M6):
    without it, the profiler could flag severe imbalance and nothing in the
    plan could act on that observation. Restricted to "balanced" (the only
    value every classifier template here actually accepts) rather than an
    open dict of per-class weights, matching the "typed subset, not an open
    dict" discipline the rest of this contract already follows.
    """

    n_estimators: int | None = None
    learning_rate: float | None = None
    max_depth: int | None = None
    num_leaves: int | None = None
    min_child_samples: int | None = None
    subsample: float | None = None
    colsample_bytree: float | None = None
    reg_alpha: float | None = None
    reg_lambda: float | None = None
    iterations: int | None = None
    depth: int | None = None
    l2_leaf_reg: float | None = None
    min_samples_leaf: int | None = None
    max_features: float | None = None
    C: float | None = None
    alpha: float | None = None
    n_neighbors: int | None = None
    class_weight: Literal["balanced"] | None = None

    def to_config(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ExperimentDraft(BaseModel):
    experiment_id: str
    template_id: str
    config: ModelConfigDraft = Field(default_factory=ModelConfigDraft)
    is_baseline: bool = False
    # Whitelist matches templates/registry.py's TARGET_TRANSFORMS -- the
    # grammar itself (Literal, not str) rejects any other value the LLM
    # might propose; _post_validate additionally rejects it outside
    # REGRESSION, since the type alone can't see task_type.
    target_transform: Literal["log1p"] | None = None


class ExcludedTemplateDraft(BaseModel):
    template_id: str
    reason: str


class ExperimentPlanDraft(BaseModel):
    experiments: list[ExperimentDraft] = Field(min_length=1)
    rationale: str
    # Optional-with-defaults so replaying fixtures recorded before this field
    # existed still validates.
    excluded_templates: list[ExcludedTemplateDraft] = Field(default_factory=list)


def estimate_budget(template_id: str, n_rows: int) -> BudgetEstimate:
    """Deterministic wall-time estimate; the LLM never invents budgets."""
    per_1k = _SECONDS_PER_1K_ROWS.get(template_id, 10.0)
    seconds = max(_MIN_BUDGET_SECONDS, per_1k * (n_rows / 1000.0))
    return BudgetEstimate(wall_time_seconds=round(seconds, 1))


REVISION_PROMPT = """You are the Experiment Planner of an autonomous ML \
engineering team, revising the plan after a Critic verdict of \
iterate_with_actions.

Rules:
- Propose 0-3 NEW candidate experiments that concretely address the Critic's \
actions. Do not repeat an experiment_id or an identical config that was \
already trained. Zero is a valid answer — see the stop verb below.
- template_id must come from the registry list (revisions are never baselines — \
those already exist).
- config keys must be valid model constructor parameters. Ground every choice \
in the critique actions and prior results.
- target_transform="log1p" may be set for a REGRESSION task when the Critic's \
actions, prior results, or slice_metrics point at a skewed target (e.g. a \
raw-scale MAE dominated by a few large values); never for a non-regression task.
- GROUNDING (enforced in code, not just this prompt): every experiment must cite \
at least one prior experiment_id (in cited_experiment_ids) or finding category \
(in cited_finding_categories, from fitness_findings/leakage_checks/slice_metrics) \
that motivated it. An experiment with no citation, or one citing an \
experiment_id that does not exist in this run, is DROPPED before training — \
propose it anyway only if you can actually name what motivated it.
- rationale must say in words which critique action, slice, or finding each \
experiment addresses — the citations above are the enforced, structured version \
of this same claim.
- STOP VERB: if further experiments are unlikely to move the primary metric \
meaningfully (e.g. the last two iterations barely moved it, the gap to the \
runner-up is already small, or every remaining critique action was already \
tried and did not help), set stop=true and give a concrete stop_reason instead \
of proposing more experiments. Do not stop merely because the critic has not \
yet said approve — stop only when you judge that MORE experiments specifically \
would not help.
Return only JSON matching the schema."""


class RevisionExperimentDraft(BaseModel):
    """Distinct from ExperimentDraft (the initial-plan schema): revisions are
    never baselines and must cite their evidence, which the initial plan
    (covered by excluded_templates/reasoning instead) does not require."""

    experiment_id: str
    template_id: str
    config: ModelConfigDraft = Field(default_factory=ModelConfigDraft)
    target_transform: Literal["log1p"] | None = None
    cited_experiment_ids: list[str] = Field(default_factory=list)
    cited_finding_categories: list[str] = Field(default_factory=list)
    rationale: str


class RevisionDraft(BaseModel):
    experiments: list[RevisionExperimentDraft] = Field(default_factory=list, max_length=3)
    rationale: str
    stop: bool = False
    stop_reason: str = ""


@dataclass(frozen=True)
class RevisionResult:
    """Revised candidates for one iteration. NOT an ExperimentPlan artifact —
    that contract mandates baselines-first for full plans; a revision only
    adds candidates, and its provenance flows through the trained models."""

    specs: tuple[ExperimentSpec, ...]
    rationale: str
    stop: bool = False
    stop_reason: str = ""


def _ground_revision_experiments(
    experiments: list[RevisionExperimentDraft],
    prior_experiment_ids: set[str],
    valid_finding_categories: set[str],
) -> list[RevisionExperimentDraft]:
    """Grounding rule enforced in code, not trusted from the prompt alone
    (JUDGMENT_ROADMAP.md Task 4): drop — loudly, existing convention, see
    agents/feature_executor.py's dropped_proposals — any experiment citing
    no prior evidence at all, or citing an experiment_id that does not
    exist in this run. The rest of the revision proceeds; one bad proposal
    must never sink the others."""
    accepted = []
    for exp in experiments:
        if not exp.cited_experiment_ids and not exp.cited_finding_categories:
            _log.warning(
                "revision_experiment_dropped",
                experiment_id=exp.experiment_id,
                reason="no citation of a prior experiment_id or finding category",
            )
            continue
        unknown_ids = [eid for eid in exp.cited_experiment_ids if eid not in prior_experiment_ids]
        if unknown_ids:
            _log.warning(
                "revision_experiment_dropped",
                experiment_id=exp.experiment_id,
                reason=f"cites unknown experiment_id(s): {unknown_ids}",
            )
            continue
        unknown_categories = [
            c for c in exp.cited_finding_categories if c not in valid_finding_categories
        ]
        if exp.cited_finding_categories and unknown_categories and not exp.cited_experiment_ids:
            _log.warning(
                "revision_experiment_dropped",
                experiment_id=exp.experiment_id,
                reason=f"cites unknown finding category(ies): {unknown_categories}",
            )
            continue
        accepted.append(exp)
    return accepted


def revise_experiment_plan(
    client: LLMClient,
    *,
    problem: ProblemSpec,
    critique_actions: tuple[str, ...],
    prior_results: list[dict[str, Any]],
    prior_experiment_ids: set[str],
    n_rows: int,
    slice_metrics: list[dict[str, Any]] | None = None,
    leakage_checks: list[dict[str, Any]] | None = None,
    fitness_findings: list[dict[str, Any]] | None = None,
) -> RevisionResult:
    """Ask the planner for 0-3 new candidates addressing the critique, or an
    explicit stop.

    ``slice_metrics``/``leakage_checks``/``fitness_findings``, when given,
    are the evidence that already exists at iteration time (JUDGMENT_
    ROADMAP.md Task 4) — deliberately NOT a fresh OOF computation for
    regression (that cost was removed 2026-07-08); these are read straight
    off the EvaluationReport/CritiqueReport the caller already built.
    """
    registry = [
        {"template_id": t.template_id, "description": t.description}
        for t in list_templates(problem.task_type)
    ]
    registry_ids = {entry["template_id"] for entry in registry}
    valid_finding_categories = {f["category"] for f in (fitness_findings or [])} | {
        c["name"] for c in (leakage_checks or [])
    }

    def _post_validate(draft: RevisionDraft) -> None:
        for exp in draft.experiments:
            if exp.template_id not in registry_ids:
                raise ValueError(
                    f"template '{exp.template_id}' is not in the registry; "
                    f"choose from: {sorted(registry_ids)}"
                )
            if exp.experiment_id in prior_experiment_ids:
                raise ValueError(
                    f"experiment_id '{exp.experiment_id}' was already used; "
                    "pick a new id for the revised experiment"
                )
            if exp.target_transform is not None and problem.task_type is not TaskType.REGRESSION:
                raise ValueError(
                    f"target_transform is only valid for regression; "
                    f"experiment '{exp.experiment_id}' is {problem.task_type.value}"
                )

    result = client.structured(
        agent="experiment-planner-revision",
        system_prompt=REVISION_PROMPT,
        user_payload={
            "critique_actions": list(critique_actions),
            "prior_results": prior_results,
            "template_registry": registry,
            "used_experiment_ids": sorted(prior_experiment_ids),
            "slice_metrics": slice_metrics or [],
            "leakage_checks": leakage_checks or [],
            "fitness_findings": fitness_findings or [],
        },
        schema=RevisionDraft,
        post_validate=_post_validate,
    )
    accepted = _ground_revision_experiments(
        result.output.experiments, prior_experiment_ids, valid_finding_categories
    )
    return RevisionResult(
        specs=tuple(
            ExperimentSpec(
                experiment_id=e.experiment_id,
                template_id=e.template_id,
                config=e.config.to_config(),
                budget=estimate_budget(e.template_id, n_rows),
                is_baseline=False,
                target_transform=e.target_transform,
            )
            for e in accepted
        ),
        rationale=result.output.rationale,
        stop=result.output.stop,
        stop_reason=result.output.stop_reason,
    )


def plan_experiments(
    client: LLMClient,
    store: ArtifactStore,
    *,
    profile: DataProfile,
    problem: ProblemSpec,
    n_rows: int,
    run_id: str,
    store_root: Path | None = None,
    screening_results: list[dict[str, Any]] | None = None,
) -> ExperimentPlan:
    """Draft, validate, and store an ExperimentPlan (unapproved until Gate 2).

    ``store_root``, when given, enables cross-run priors: up to 5 past runs
    on similarly-sized data of the same task type are read from the outcome
    ledger and included in the LLM payload so the planner can reorder
    candidates on experience — never to skip baselines or coverage, which
    stay code-enforced by the plan validator regardless of what the priors say.

    ``screening_results``, when given, are this run's own cheap probes
    (dummy/linear/lightgbm, already trained) — real evidence from THIS
    dataset, stronger than any cross-run prior, that conditions the real
    candidate slate instead of planning blind.
    """
    registry = [
        {"template_id": t.template_id, "description": t.description}
        for t in list_templates(problem.task_type)
    ]
    registry_ids = {entry["template_id"] for entry in registry}

    def _post_validate(draft: ExperimentPlanDraft) -> None:
        ids = [e.experiment_id for e in draft.experiments]
        if len(ids) != len(set(ids)):
            raise ValueError("experiment_id values must be unique")
        for exp in draft.experiments:
            if exp.template_id not in registry_ids:
                raise ValueError(
                    f"template '{exp.template_id}' is not in the registry; "
                    f"choose from: {sorted(registry_ids)}"
                )
            if not get_template(exp.template_id).supports(problem.task_type):
                raise ValueError(
                    f"template '{exp.template_id}' does not support {problem.task_type.value}"
                )
            if (
                exp.config.class_weight is not None
                and exp.template_id not in _CLASS_WEIGHT_CAPABLE_TEMPLATES
            ):
                raise ValueError(
                    f"template '{exp.template_id}' does not accept class_weight; "
                    f"only {sorted(_CLASS_WEIGHT_CAPABLE_TEMPLATES)} do"
                )
            if exp.target_transform is not None and problem.task_type is not TaskType.REGRESSION:
                raise ValueError(
                    f"target_transform is only valid for regression; "
                    f"experiment '{exp.experiment_id}' is {problem.task_type.value}"
                )
        first_two = [e.template_id for e in draft.experiments[:2]]
        if first_two[:1] != ["dummy_baseline"] or (
            len(first_two) < 2 or first_two[1] != "lightgbm_default"
        ):
            raise ValueError(
                "the plan must start with dummy_baseline then lightgbm_default, "
                "both with is_baseline=true"
            )
        if not all(e.is_baseline for e in draft.experiments[:2]):
            raise ValueError("the first two experiments must have is_baseline=true")

    priors = (
        matching_priors(
            store_root,
            task_type=problem.task_type.value,
            n_rows=n_rows,
            primary_metric=problem.primary_metric,
        )
        if store_root is not None
        else []
    )
    lessons = (
        matching_lessons(store_root, task_type=problem.task_type.value, max_entries=_LESSONS_TOP_K)
        if store_root is not None
        else []
    )

    result = client.structured(
        agent="experiment-planner",
        system_prompt=SYSTEM_PROMPT,
        user_payload={
            "problem": {
                "task_type": problem.task_type.value,
                "primary_metric": problem.primary_metric,
            },
            "n_rows": n_rows,
            "template_registry": registry,
            "profile": profile_summary(profile),
            "prior_outcomes": [
                {
                    "winner_template": p["winner_template"],
                    "margin_vs_second": p["margin_vs_second"],
                    "n_features": p["n_features"],
                    "categorical_share": p["categorical_share"],
                }
                for p in priors
            ],
            "screening_results": screening_results or [],
            "lessons_from_previous_runs": [
                {"lesson": entry["lesson"], "evidence": entry["evidence"]} for entry in lessons
            ],
        },
        schema=ExperimentPlanDraft,
        post_validate=_post_validate,
    )
    draft = result.output
    excluded = tuple(
        ExcludedTemplate(template_id=e.template_id, reason=e.reason)
        for e in draft.excluded_templates
    )
    plan = new_artifact(
        ExperimentPlan,
        producer_agent="experiment-planner",
        run_id=run_id,
        parent_ids=(profile.meta.artifact_id, problem.meta.artifact_id),
        experiments=tuple(
            ExperimentSpec(
                experiment_id=e.experiment_id,
                template_id=e.template_id,
                config=e.config.to_config(),
                budget=estimate_budget(e.template_id, n_rows),
                is_baseline=e.is_baseline,
                target_transform=e.target_transform,
            )
            for e in draft.experiments
        ),
        rationale=draft.rationale,
        human_approved=False,
        excluded_templates=excluded,
        reasoning=DecisionRecord(
            decision=f"templates: {[e.template_id for e in draft.experiments]}",
            why=draft.rationale,
            alternatives_considered=tuple(f"{e.template_id}: {e.reason}" for e in excluded),
        ),
    )
    store.put(plan)
    return plan
