"""Benchmark dataset manifest.

Seven datasets spanning binary / multiclass / regression, including one
heavily imbalanced (sick: ~6% positive) and one with a real datetime column
(temporal_churn, generated deterministically). OpenML ids are pinned so the
data never shifts under us.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlteam.contracts import TaskType


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    task_type: TaskType
    target_column: str
    primary_metric: str
    source: str  # "openml" | "synthetic"
    openml_id: int | None = None
    imbalanced: bool = False
    has_datetime: bool = False


BENCHMARKS: tuple[BenchmarkDataset, ...] = (
    BenchmarkDataset(
        name="credit_g",
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="class",
        primary_metric="roc_auc",
        source="openml",
        openml_id=31,
    ),
    BenchmarkDataset(
        name="diabetes",
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="class",
        primary_metric="roc_auc",
        source="openml",
        openml_id=37,
    ),
    BenchmarkDataset(
        name="sick",
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="Class",
        primary_metric="roc_auc",
        source="openml",
        openml_id=38,
        imbalanced=True,
    ),
    BenchmarkDataset(
        name="vehicle",
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        target_column="Class",
        primary_metric="accuracy",
        source="openml",
        openml_id=54,
    ),
    BenchmarkDataset(
        name="cmc",
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        target_column="Contraceptive_method_used",
        primary_metric="accuracy",
        source="openml",
        openml_id=23,
    ),
    BenchmarkDataset(
        name="cpu_small",
        task_type=TaskType.REGRESSION,
        target_column="usr",
        primary_metric="rmse",
        source="openml",
        openml_id=227,
    ),
    BenchmarkDataset(
        name="temporal_churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churned",
        primary_metric="roc_auc",
        source="synthetic",
        has_datetime=True,
    ),
)


def get_benchmark(name: str) -> BenchmarkDataset:
    for entry in BENCHMARKS:
        if entry.name == name:
            return entry
    raise KeyError(f"unknown benchmark '{name}'; known: {[b.name for b in BENCHMARKS]}")
