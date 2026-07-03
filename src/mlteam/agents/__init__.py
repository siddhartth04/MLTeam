"""Agents: pure functions over artifacts. No agent talks to another directly."""

from mlteam.agents.cleaner import execute_cleaning, propose_cleaning_ops
from mlteam.agents.data_io import read_dataset, store_dataframe
from mlteam.agents.ingest import ingest_file
from mlteam.agents.profiler import profile_dataset
from mlteam.agents.splitter import plan_split
from mlteam.agents.trainer import TrainResult, compute_env_hash, train_experiment

__all__ = [
    "TrainResult",
    "compute_env_hash",
    "execute_cleaning",
    "ingest_file",
    "plan_split",
    "profile_dataset",
    "propose_cleaning_ops",
    "read_dataset",
    "store_dataframe",
    "train_experiment",
]
