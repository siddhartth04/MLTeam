"""Agents: pure functions over artifacts. No agent talks to another directly."""

from mlteam.agents.cleaner import (
    EXECUTABLE_OP_TYPES,
    execute_cleaning,
    propose_cleaning_ops,
    propose_cleaning_ops_llm,
)
from mlteam.agents.critic import compute_quantitative_gates, critique_evaluation
from mlteam.agents.data_io import read_dataset, store_dataframe
from mlteam.agents.experiment_planner import (
    estimate_budget,
    plan_experiments,
    revise_experiment_plan,
)
from mlteam.agents.feature_engineer import propose_features
from mlteam.agents.hpo import SEARCH_SPACES, HPOOutcome, tune_experiment
from mlteam.agents.ingest import ingest_file
from mlteam.agents.profiler import profile_dataset
from mlteam.agents.requirement import understand_requirements
from mlteam.agents.splitter import plan_split
from mlteam.agents.trainer import TrainResult, compute_env_hash, train_experiment

__all__ = [
    "EXECUTABLE_OP_TYPES",
    "SEARCH_SPACES",
    "HPOOutcome",
    "TrainResult",
    "compute_env_hash",
    "compute_quantitative_gates",
    "critique_evaluation",
    "estimate_budget",
    "execute_cleaning",
    "ingest_file",
    "plan_experiments",
    "plan_split",
    "profile_dataset",
    "propose_cleaning_ops",
    "propose_cleaning_ops_llm",
    "propose_features",
    "read_dataset",
    "revise_experiment_plan",
    "store_dataframe",
    "train_experiment",
    "tune_experiment",
    "understand_requirements",
]
