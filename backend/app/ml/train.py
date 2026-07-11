"""
Train the Gradient Boosting rep-prediction model.

The fitted artifact is a scikit-learn Pipeline that takes a DataFrame with
``FEATURE_COLUMNS`` (exercise as a string) and returns predicted reps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from app.features import EXERCISES, FEATURE_COLUMNS, NUMERIC_FEATURES
from app.ml.generate_data import generate_dataset

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


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="CSV from generate_data (generated if omitted)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.data) if args.data else generate_dataset(seed=args.seed)
    result = train(df)
    print(f"MAE {result.metrics['mae']:.3f}  RMSE {result.metrics['rmse']:.3f}  R2 {result.metrics['r2']:.4f}")
    for name, imp in sorted(result.feature_importance.items(), key=lambda kv: -kv[1]):
        print(f"  {name:20s} {imp:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, args.out)
    print(f"Model saved -> {args.out}")


if __name__ == "__main__":
    main()
