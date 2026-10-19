"""MLflow setup shared by the EDA and training scripts."""
from __future__ import annotations

from pathlib import Path

import mlflow

from . import config


def setup_mlflow(db: Path = config.MLFLOW_DB, artifacts: Path = config.MLFLOW_ARTIFACTS) -> str:
    """Point MLflow at a local SQLite store and return the experiment id.

    A SQLite backend (instead of the deprecated plain-file store) gives us the
    model registry for free and works the same on a laptop and in CI.
    """
    db, artifacts = Path(db), Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if exp is None:
        return mlflow.create_experiment(config.MLFLOW_EXPERIMENT, artifact_location=artifacts.resolve().as_uri())
    return exp.experiment_id
