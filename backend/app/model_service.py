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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import mlflow.sklearn
from mlflow import MlflowClient

from app.features import ModelInput, to_frame
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


@dataclass(frozen=True)
class LoadedModel:
    ref: ModelRef
    model: Any
    loaded_at: datetime


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
            self._current = LoadedModel(ref=ref, model=model, loaded_at=datetime.now(UTC))
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

    def predict_with_version(self, inputs: Sequence[ModelInput]) -> tuple[list[float], ModelRef]:
        snapshot = self._current
        if snapshot is None:
            raise ModelUnavailableError("no model loaded")
        preds = snapshot.model.predict(to_frame(inputs))
        return [float(v) for v in preds], snapshot.ref

    def predict(self, inputs: Sequence[ModelInput]) -> list[float]:
        return self.predict_with_version(inputs)[0]
