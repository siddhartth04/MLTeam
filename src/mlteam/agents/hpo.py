"""Hyperparameter optimization: Optuna with successive-halving pruning.

Budget-aware (n_trials + wall timeout), deterministic (seeded TPE sampler,
single-threaded), leakage-safe (the objective cross-validates full pipelines —
preprocessing refit inside every fold). The winning config is retrained and
stored via the ordinary trainer, so HPO output is a first-class TrainedModel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

from mlteam.agents.data_io import read_dataset
from mlteam.agents.trainer import (
    TrainResult,
    _column_roles,
    _make_cv,
    _temporal_to_epoch,
    train_experiment,
)
from mlteam.contracts import BudgetEstimate, ExperimentSpec, TaskType
from mlteam.evaluation.metrics import scoring_for, to_natural_sign
from mlteam.store.interface import ArtifactStore
from mlteam.templates.registry import get_template, wrap_target_transform

_log = structlog.get_logger(__name__)

SearchSpace = Callable[[Any], dict[str, Any]]  # optuna.Trial -> params


def lightgbm_search_space(trial: Any) -> dict[str, Any]:
    """Vetted LightGBM search space — the LLM never invents ranges."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def xgboost_search_space(trial: Any) -> dict[str, Any]:
    """Vetted XGBoost search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def catboost_search_space(trial: Any) -> dict[str, Any]:
    """Vetted CatBoost search space."""
    return {
        "iterations": trial.suggest_int("iterations", 100, 500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }


def random_forest_search_space(trial: Any) -> dict[str, Any]:
    """Vetted random-forest search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


SEARCH_SPACES: dict[str, SearchSpace] = {
    "lightgbm_default": lightgbm_search_space,
    "xgboost_default": xgboost_search_space,
    "catboost_default": catboost_search_space,
    "random_forest_default": random_forest_search_space,
    "extra_trees_default": random_forest_search_space,  # same knob semantics
}


@dataclass(frozen=True)
class HPOOutcome:
    result: TrainResult
    best_params: dict[str, Any]
    n_trials_run: int
    n_trials_pruned: int
    # Optuna's own best CV score, from the SAME folds (seed) the search
    # optimized against -- upward-biased (the max of many noisy trial
    # estimates is not an unbiased estimate of any one config's true
    # performance; Cawley & Talbot, 2010). Kept for observability/debugging,
    # NOT for comparison against other experiments' scores.
    search_cv_score: float
    # The winning config's CV score on report_seed's folds (the SAME folds
    # every other experiment this run was scored on) -- an independent
    # re-evaluation of the config the search already fixed, restoring both
    # (a) an honest score not selected-for by the search that produced it,
    # and (b) fold pairing with incumbent experiments for significance
    # testing (evaluation/stats.py's paired Nadeau-Bengio test requires the
    # SAME fold assignment on both sides of a comparison).
    reported_cv_score: float

    @property
    def best_cv_score(self) -> float:
        """Deprecated alias for reported_cv_score -- kept so existing
        callers reading .best_cv_score get the CORRECTED (re-scored) value,
        not the search's biased estimate, without needing to change."""
        return self.reported_cv_score


def _objective_factory(
    frame: Any,
    target: Any,
    *,
    template_id: str,
    task_type: TaskType,
    roles: Any,
    primary_metric: str,
    n_folds: int,
    seed: int,
    space: SearchSpace,
    target_transform: str | None = None,
) -> Callable[[Any], float]:
    """Objective with fold-by-fold reporting so ASHA pruning can bite.

    sklearn scorers are maximize-oriented by convention (neg_rmse etc. already
    return higher-is-better values), so the study maximizes the raw scorer
    value with NO manual sign flip — the study direction is always 'maximize'.

    ``target_transform``, when set, wraps each trial's pipeline the same way
    the deterministic trainer does — the search must optimize on the SAME
    scale the winning model will actually be scored/served on, or a trial
    that looks good under log1p could lose to one that only looks good
    raw.
    """
    import optuna
    from sklearn.base import clone
    from sklearn.metrics import get_scorer

    template = get_template(template_id)
    scorer = get_scorer(scoring_for(task_type)[primary_metric])
    cv = _make_cv(task_type, n_folds, seed)

    def objective(trial: Any) -> float:
        params = space(trial)
        pipeline = wrap_target_transform(
            template.build(task_type, roles, params, seed), target_transform, task_type
        )
        scores: list[float] = []
        for step, (train_idx, test_idx) in enumerate(cv.split(frame, target)):
            fitted = clone(pipeline).fit(frame.iloc[train_idx], target.iloc[train_idx])
            fold_score = float(scorer(fitted, frame.iloc[test_idx], target.iloc[test_idx]))
            scores.append(fold_score)
            trial.report(sum(scores) / len(scores), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return sum(scores) / len(scores)

    return objective


def tune_experiment(
    store: ArtifactStore,
    dataset_id: str,
    *,
    template_id: str,
    task_type: TaskType,
    primary_metric: str,
    run_id: str,
    seed: int,
    n_folds: int,
    env_hash: str,
    n_trials: int = 20,
    timeout_seconds: float | None = None,
    space: SearchSpace | None = None,
    experiment_id: str | None = None,
    mlflow_enabled: bool = False,
    report_seed: int | None = None,
    target_transform: str | None = None,
) -> HPOOutcome:
    """Run an Optuna study and store the retrained best config as a model.

    ``seed`` drives the search (decorrelating repeated tuning rounds across
    iterations is fine and even desirable there). ``report_seed`` — when
    different from ``seed`` — is the fold assignment the FINAL model is
    trained and scored on for reporting/selection: re-scoring on folds the
    search did not see removes most of the search's selection bias (Cawley
    & Talbot, 2010) and restores fold pairing with whatever other
    experiments this run also scored on ``report_seed``'s folds. Defaults
    to ``seed`` (the old behavior) when omitted.
    """
    import optuna

    report_seed = seed if report_seed is None else report_seed

    space = space if space is not None else SEARCH_SPACES.get(template_id)
    if space is None:
        raise ValueError(
            f"template '{template_id}' has no vetted search space; "
            f"tunable templates: {sorted(SEARCH_SPACES)}"
        )

    dataset = store.get(dataset_id)
    if dataset.target_column is None:
        raise ValueError("HPO dataset has no declared target column")
    df = _temporal_to_epoch(read_dataset(store, dataset))
    roles = _column_roles(df, dataset.target_column)
    frame = df.to_pandas()
    target = frame.pop(dataset.target_column)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )
    study.optimize(
        _objective_factory(
            frame,
            target,
            template_id=template_id,
            task_type=task_type,
            roles=roles,
            primary_metric=primary_metric,
            n_folds=n_folds,
            seed=seed,
            space=space,
            target_transform=target_transform,
        ),
        n_trials=n_trials,
        timeout=timeout_seconds,
        n_jobs=1,
        show_progress_bar=False,
    )
    pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    # study.best_value is the raw maximize-oriented scorer value (e.g. -rmse);
    # report it in the metric's natural sign for humans and downstream code.
    # This is the search's OWN estimate on the folds it optimized against --
    # upward-biased, kept only for observability (see search_cv_score below).
    scorer_name = scoring_for(task_type)[primary_metric]
    search_score_natural = to_natural_sign(primary_metric, scorer_name, [study.best_value])[0]
    _log.info(
        "hpo_study_complete",
        template=template_id,
        search_score=round(search_score_natural, 6),
        best_params=study.best_params,
        trials=len(study.trials),
        pruned=pruned,
    )

    spec = ExperimentSpec(
        experiment_id=experiment_id or f"hpo-{template_id}",
        template_id=template_id,
        config=dict(study.best_params),
        budget=BudgetEstimate(wall_time_seconds=0.0),
        is_baseline=False,
        target_transform=target_transform,
    )
    # Re-score the winning config on report_seed's folds -- NOT the folds
    # the search just optimized it against. When report_seed == seed (the
    # default) this is unchanged from the old behavior; when they differ,
    # this single extra CV removes most of the search's selection bias and
    # restores fold pairing with whatever else this run was scored on
    # report_seed's folds.
    result = train_experiment(
        store,
        dataset_id,
        spec,
        task_type=task_type,
        run_id=run_id,
        seed=report_seed,
        n_folds=n_folds,
        env_hash=env_hash,
        mlflow_enabled=mlflow_enabled,
    )
    reported_score = result.metrics[primary_metric].mean
    if report_seed != seed:
        _log.info(
            "hpo_winner_rescored",
            template=template_id,
            search_score=round(search_score_natural, 6),
            reported_score=round(reported_score, 6),
            search_seed=seed,
            report_seed=report_seed,
        )
    return HPOOutcome(
        result=result,
        best_params=dict(study.best_params),
        n_trials_run=len(study.trials),
        n_trials_pruned=pruned,
        search_cv_score=float(search_score_natural),
        reported_cv_score=float(reported_score),
    )
