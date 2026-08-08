"""Cached inference and cached history-feature lookups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.orm import Session

from app import repository
from app.cache import Cache, features_key, prediction_key
from app.features import ModelInput, TrainingContext, compute_context
from app.ml import predictor
from app.model_service import LoadedModel, ModelService

PREDICTION_CACHE = "prediction"
FEATURES_CACHE = "features"


@dataclass(frozen=True)
class PredictionResult:
    raw: float
    reps: int
    outcome: predictor.Outcome
    model_version: int
    cached: bool


@dataclass(frozen=True)
class HistoryFeatures:
    """Everything the recommendation/alert endpoints need from the DB for one exercise."""

    context: TrainingContext | None
    top_weights: list[float]
    top_rirs: list[int]

    def to_json(self) -> dict[str, Any]:
        return {
            "context": asdict(self.context) if self.context else None,
            "top_weights": self.top_weights,
            "top_rirs": self.top_rirs,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> HistoryFeatures:
        ctx = data["context"]
        return cls(
            context=TrainingContext(**ctx) if ctx else None,
            top_weights=list(data["top_weights"]),
            top_rirs=list(data["top_rirs"]),
        )


class _PinnedModel:
    """RepModel over one model snapshot, so a multi-call recommendation uses a single version."""

    def __init__(self, inference: Inference, loaded: LoadedModel) -> None:
        self._inference = inference
        self._loaded = loaded

    def predict(self, inputs: Sequence[ModelInput]) -> list[float]:
        return [self._inference.predict_with(self._loaded, i).raw for i in inputs]


class Inference:
    def __init__(
        self,
        models: ModelService,
        cache: Cache,
        prediction_ttl_seconds: int,
        feature_ttl_seconds: int,
    ) -> None:
        self.models = models
        self.cache = cache
        self.prediction_ttl_seconds = prediction_ttl_seconds
        self.feature_ttl_seconds = feature_ttl_seconds

    def predict(self, features: ModelInput) -> PredictionResult:
        return self.predict_with(self.models.require(), features)

    def predict_with(self, loaded: LoadedModel, features: ModelInput) -> PredictionResult:
        key = prediction_key(loaded.ref.version, features.as_row())
        cached = self.cache.get_json(PREDICTION_CACHE, key)
        if cached is not None:
            raw = float(cached)
        else:
            raw = loaded.predict([features])[0]
            self.cache.set_json(key, raw, self.prediction_ttl_seconds)
        reps = predictor.round_reps(raw)
        return PredictionResult(
            raw=raw,
            reps=reps,
            outcome=predictor.outcome_for(reps),
            model_version=loaded.ref.version,
            cached=cached is not None,
        )

    def pinned_model(self) -> _PinnedModel:
        return _PinnedModel(self, self.models.require())

    def history_features(self, db: Session, exercise: str) -> HistoryFeatures:
        key = features_key(exercise)
        cached = self.cache.get_json(FEATURES_CACHE, key)
        if cached is not None:
            return HistoryFeatures.from_json(cached)
        history = repository.session_history(db, exercise)
        result = HistoryFeatures(
            context=compute_context(history, exercise),
            top_weights=[s.top_weight for s in history],
            top_rirs=[s.top_rir for s in history],
        )
        self.cache.set_json(key, result.to_json(), self.feature_ttl_seconds)
        return result

    def invalidate_features(self, exercise: str) -> None:
        self.cache.delete(features_key(exercise))
