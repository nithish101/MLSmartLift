"""
Serves predictions from the model registered in MLflow under a given stage.

The service resolves ``models:/<name>`` at ``<stage>`` at startup and then polls
the registry; when a different version is promoted into the stage it loads that
version and swaps it in atomically, so promotions take effect without a redeploy.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

from app.features import NUMERIC_FEATURES, ModelInput, to_frame
from app.metrics import MODEL_RELOADS, MODEL_VERSION
from app.registry import ModelRef, latest_in_stage

logger = logging.getLogger(__name__)

# MLflow's default HTTP retry policy backs off for minutes when the registry is
# down, which would stall startup; fail fast instead and let the poller retry.
_MLFLOW_HTTP_DEFAULTS = {
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": "1",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
}


class ModelUnavailableError(RuntimeError):
    pass


FastPredict = Callable[[Sequence[ModelInput]], Any]


def fast_path(model: Any) -> FastPredict | None:
    """
    Equivalent of ``model.predict(to_frame(inputs))`` that skips the DataFrame and
    ColumnTransformer: on one row their overhead is ~8x the cost of the trees.

    Only built when the pipeline has exactly the structure training produces
    (ordinal-encoded exercise, then the numeric features passed through);
    anything else returns None and serving uses the pipeline as-is.
    """
    try:
        transformer = model.named_steps["features"]
        estimator = model.named_steps["model"]
        fitted = [(n, t, list(c)) for n, t, c in transformer.transformers_ if not (n == "remainder" and t == "drop")]
        (enc_name, encoder, enc_cols), (num_name, passthrough, num_cols) = fitted
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    # Fitted ColumnTransformers replace "passthrough" with an identity FunctionTransformer.
    identity = passthrough == "passthrough" or (
        type(passthrough).__name__ == "FunctionTransformer" and passthrough.func is None
    )
    if not identity or (enc_name, enc_cols, num_name, num_cols) != (
        "exercise",
        ["exercise"],
        "numeric",
        NUMERIC_FEATURES,
    ):
        return None
    codes = {category: float(i) for i, category in enumerate(encoder.categories_[0])}

    def predict(inputs: Sequence[ModelInput]) -> Any:
        rows = [[codes[i.exercise], *(float(getattr(i, c)) for c in NUMERIC_FEATURES)] for i in inputs]
        return estimator.predict(np.asarray(rows, dtype=float))

    return predict


@dataclass(frozen=True)
class LoadedModel:
    ref: ModelRef
    model: Any
    loaded_at: datetime
    fast: FastPredict | None = None

    def predict(self, inputs: Sequence[ModelInput]) -> list[float]:
        preds = self.fast(inputs) if self.fast is not None else self.model.predict(to_frame(inputs))
        return [float(v) for v in preds]


class ModelService:
    def __init__(self, tracking_uri: str, model_name: str, stage: str) -> None:
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.stage = stage
        for key, value in _MLFLOW_HTTP_DEFAULTS.items():
            os.environ.setdefault(key, value)
        self._client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        self._current: LoadedModel | None = None
        self._refresh_lock = threading.Lock()
        MODEL_VERSION.labels(model_name, stage).set(0)

    @property
    def current(self) -> LoadedModel | None:
        return self._current

    @property
    def loaded(self) -> bool:
        return self._current is not None

    def refresh(self) -> bool:
        """Load the stage's current version if it changed. Returns True on a swap."""
        with self._refresh_lock:
            ref = latest_in_stage(self._client, self.model_name, self.stage)
            if ref is None:
                logger.warning("no version of %s in stage %s", self.model_name, self.stage)
                return False
            current = self._current
            if current is not None and current.ref.version == ref.version:
                return False
            mlflow.set_tracking_uri(self.tracking_uri)
            model = mlflow.sklearn.load_model(f"models:/{ref.name}/{ref.version}")
            self._current = LoadedModel(ref=ref, model=model, loaded_at=datetime.now(UTC), fast=fast_path(model))
            MODEL_VERSION.labels(self.model_name, self.stage).set(ref.version)
            MODEL_RELOADS.inc()
            logger.info(
                "loaded model %s v%s (%s)",
                ref.name,
                ref.version,
                ref.stage,
                extra={"model_version": ref.version, "previous_version": current.ref.version if current else None},
            )
            return True

    def safe_refresh(self) -> bool:
        try:
            return self.refresh()
        except Exception:
            logger.exception("model refresh failed; keeping current model")
            return False

    async def poll(self, interval_seconds: float) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            await asyncio.to_thread(self.safe_refresh)

    def require(self) -> LoadedModel:
        """The current model snapshot; raises if none is loaded."""
        snapshot = self._current
        if snapshot is None:
            raise ModelUnavailableError("no model loaded")
        return snapshot

    def predict(self, inputs: Sequence[ModelInput]) -> list[float]:
        return self.require().predict(inputs)
