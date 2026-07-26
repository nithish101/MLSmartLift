"""
Train the Gradient Boosting rep-prediction model and register it in MLflow.

The fitted artifact is a scikit-learn Pipeline that takes a DataFrame with
``FEATURE_COLUMNS`` (exercise as a string) and returns predicted reps. Each run
logs its parameters, MAE/RMSE/R2 and the model to the tracking server, and
registers the model as a new version of ``MODEL_NAME``.

    python -m app.ml.train                      # train + register (no stage)
    python -m app.ml.train --stage Staging      # ... and move it to Staging
    python -m app.ml.train --bootstrap          # only if nothing is in Production yet
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.preprocessing import OrdinalEncoder

from app.features import EXERCISES, FEATURE_COLUMNS, NUMERIC_FEATURES
from app.ml.generate_data import generate_dataset
from app.registry import STAGES, ModelRef, latest_in_stage, promote

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "subsample": 0.8,
    "random_state": 42,
}


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: dict[str, float]
    params: dict[str, Any]
    feature_importance: dict[str, float]


def build_pipeline(params: dict[str, Any]) -> Pipeline:
    encoder = ColumnTransformer(
        [
            (
                "exercise",
                OrdinalEncoder(categories=[list(EXERCISES)]),
                ["exercise"],
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("features", encoder), ("model", GradientBoostingRegressor(**params))])


def train(df: pd.DataFrame, params: dict[str, Any] | None = None, test_size: float = 0.2) -> TrainResult:
    params = {**DEFAULT_PARAMS, **(params or {})}
    X = df[FEATURE_COLUMNS]
    y = df["reps"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    pipeline = build_pipeline(params)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
    model: GradientBoostingRegressor = pipeline.named_steps["model"]
    importance = dict(zip(FEATURE_COLUMNS, (float(v) for v in model.feature_importances_), strict=True))
    return TrainResult(pipeline=pipeline, metrics=metrics, params=params, feature_importance=importance)


def log_and_register(
    result: TrainResult,
    df: pd.DataFrame,
    *,
    tracking_uri: str,
    experiment: str,
    model_name: str,
    extra_params: dict[str, Any] | None = None,
) -> ModelRef:
    """Log a training run to MLflow and register its model as a new version."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    # Numeric features are declared as doubles so the signature tolerates either ints or floats.
    sample = df[FEATURE_COLUMNS].head(20).astype({c: "float64" for c in NUMERIC_FEATURES})
    with mlflow.start_run() as run:
        mlflow.log_params({**result.params, "n_samples": len(df), **(extra_params or {})})
        mlflow.log_metrics(result.metrics)
        mlflow.log_metrics({f"importance_{k}": v for k, v in result.feature_importance.items()})
        info = mlflow.sklearn.log_model(
            result.pipeline,
            name="model",
            signature=infer_signature(sample, result.pipeline.predict(sample)),
            input_example=sample.head(3),
            registered_model_name=model_name,
        )
    version = info.registered_model_version
    if version is None:  # pragma: no cover - registration is requested above
        raise RuntimeError("model was not registered")
    return ModelRef(name=model_name, version=int(version), stage="None", run_id=run.info.run_id)


def run_training(
    *,
    tracking_uri: str,
    experiment: str,
    model_name: str,
    stage: str | None = None,
    bootstrap: bool = False,
    data: Path | None = None,
    num_lifters: int = 50,
    seed: int = 0,
    params: dict[str, Any] | None = None,
) -> ModelRef | None:
    """Train, register and optionally promote. Returns None if bootstrap skipped training."""
    client = MlflowClient(tracking_uri=tracking_uri)
    if bootstrap:
        existing = latest_in_stage(client, model_name, "Production")
        if existing is not None:
            logger.info("%s v%s already in Production; skipping bootstrap", model_name, existing.version)
            return None
        stage = "Production"

    df = pd.read_csv(data) if data else generate_dataset(num_lifters=num_lifters, seed=seed)
    result = train(df, params=params)
    logger.info("trained: MAE %.3f RMSE %.3f R2 %.4f", result.metrics["mae"], result.metrics["rmse"], result.metrics["r2"])

    ref = log_and_register(
        result,
        df,
        tracking_uri=tracking_uri,
        experiment=experiment,
        model_name=model_name,
        extra_params={"seed": seed, "num_lifters": num_lifters, "data": str(data) if data else "synthetic"},
    )
    if stage:
        ref = promote(client, model_name, ref.version, stage)
    logger.info("registered %s v%s (stage %s)", ref.name, ref.version, ref.stage)
    return ref


def main(argv: Sequence[str] | None = None) -> None:
    from app.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None, help="CSV from generate_data (synthetic if omitted)")
    parser.add_argument("--lifters", type=int, default=50, help="synthetic lifters to generate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=STAGES, default=None, help="promote the new version to this stage")
    parser.add_argument("--bootstrap", action="store_true", help="train only if no version is in Production")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_training(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment=settings.mlflow_experiment,
        model_name=settings.model_name,
        stage=args.stage,
        bootstrap=args.bootstrap,
        data=args.data,
        num_lifters=args.lifters,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
