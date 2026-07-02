"""MLflow wiring: local file backend, one experiment per project run.

MLflow import is deferred inside functions — it is a heavy import and Phase 0
callers (tests, contracts) should not pay for it unless tracking is used.
"""

from __future__ import annotations

from typing import Any

import structlog

_log = structlog.get_logger(__name__)


def setup_mlflow(tracking_uri: str, experiment_name: str = "mlteam") -> str:
    """Point MLflow at the local backend and ensure the experiment exists.

    Returns the experiment id.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    _log.info("mlflow_ready", tracking_uri=tracking_uri, experiment_id=experiment_id)
    return experiment_id


def log_run_context(
    run_id: str,
    config: dict[str, Any],
    data_hash: str | None = None,
) -> None:
    """Attach mlteam run identity, config, and data hash to the active MLflow run."""
    import mlflow

    mlflow.set_tag("mlteam.run_id", run_id)
    if data_hash is not None:
        mlflow.set_tag("mlteam.data_hash", data_hash)
    for key, value in config.items():
        mlflow.log_param(key, value)
