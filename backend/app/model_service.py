"""Loads the fitted model and serves predictions."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib

from app.features import ModelInput, to_frame

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class ModelService:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self._model: Any | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            logger.warning("model artifact not found at %s; predictions disabled", self.model_path)
            return
        self._model = joblib.load(self.model_path)
        logger.info("loaded model from %s", self.model_path)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def predict(self, inputs: Sequence[ModelInput]) -> list[float]:
        if self._model is None:
            raise ModelUnavailableError("no model loaded")
        return [float(v) for v in self._model.predict(to_frame(inputs))]
