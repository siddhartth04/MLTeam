"""End-to-end Phase 1 example: churn dataset through the Prefect flow.

Run:  uv run python examples/run_example.py
(Generates the dataset first if missing. MLflow logs land in ./mlflow.db,
artifacts in ./artifact_store.)
"""

from __future__ import annotations

import os
from pathlib import Path

# Allow the flow to run without a standing Prefect server.
os.environ.setdefault("PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", "true")

from mlteam.observability import configure_logging, setup_mlflow  # noqa: E402
from mlteam.orchestration.flows import tabular_baseline_flow  # noqa: E402
from mlteam.store import open_store  # noqa: E402

DATASET = Path(__file__).parent / "customer_churn.csv"
STORE_ROOT = Path("./artifact_store")


def main() -> None:
    configure_logging(json_output=False)
    if not DATASET.exists():
        from make_example_dataset import make_churn_frame

        make_churn_frame().write_csv(DATASET)

    setup_mlflow("sqlite:///mlflow.db", experiment_name="mlteam-example")
    summary = tabular_baseline_flow(
        data_path=str(DATASET),
        target_column="churned",
        task_type="binary_classification",
        primary_metric="roc_auc",
        store_root=str(STORE_ROOT),
        mlflow_enabled=True,
    )

    store = open_store(STORE_ROOT)
    report = store.get(summary["evaluation_report_id"])
    print("\n=== Phase 1 example run complete ===")
    print(f"run_id:          {summary['run_id']}")
    print(f"best template:   {summary['best_template_id']}")
    if summary["skipped"]:
        print(f"skipped:         {summary['skipped']}")
    print(f"holdout untouched: {report.holdout_untouched}")
    print("\nCV metrics (mean +/- std):")
    for metric in report.cv_metrics:
        print(f"  {metric.name:12s} {metric.mean:8.4f} +/- {metric.std:.4f}")
    print("\nvs dummy baseline:")
    for delta in report.baseline_deltas:
        marker = "significant" if delta.significant else "not significant"
        print(
            f"  {delta.metric:12s} {delta.baseline_value:8.4f} -> "
            f"{delta.model_value:8.4f}  ({marker})"
        )
    print("\nleakage checks:")
    for check in report.leakage_checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.details}")


if __name__ == "__main__":
    main()
