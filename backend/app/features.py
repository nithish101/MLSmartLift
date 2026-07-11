"""
Feature computation for the rep-prediction model.

Two responsibilities:
  * ``ModelInput`` validates a single feature vector before it reaches the model.
  * ``compute_context`` derives the history-based features (frequency, sets,
    volume score, progression trend) from a lifter's logged sessions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

Exercise = Literal["bench_press", "deadlift", "squat"]
EXERCISES: tuple[Exercise, ...] = ("bench_press", "deadlift", "squat")

# Column order the model is trained and served with.
FEATURE_COLUMNS: list[str] = [
    "exercise",
    "weight",
    "rir",
    "bodyweight",
    "num_sets",
    "frequency",
    "volume_score",
    "progression_trend",
]
NUMERIC_FEATURES: list[str] = FEATURE_COLUMNS[1:]

MUSCLE_MULTIPLIER: dict[str, float] = {"bench_press": 1.0, "squat": 1.3, "deadlift": 1.4}

# Defaults used when a lifter has no history for an exercise.
DEFAULT_NUM_SETS = 4
DEFAULT_FREQUENCY = 3
DEFAULT_VOLUME_SCORE = 50
DEFAULT_PROGRESSION_TREND = 0.5
MAX_PROGRESSION_TREND = 1.5


class ModelInput(BaseModel):
    """One validated feature vector for the rep-prediction model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exercise: Exercise
    weight: float = Field(..., gt=0, le=1500, description="Load in lbs")
    rir: int = Field(..., ge=-1, le=4, description="Reps in reserve (-1 = failure, treated as 0)")
    bodyweight: float = Field(..., ge=50, le=700, description="Lifter bodyweight in lbs")
    num_sets: int = Field(DEFAULT_NUM_SETS, ge=1, le=20)
    frequency: int = Field(DEFAULT_FREQUENCY, ge=1, le=14, description="Sessions per week")
    volume_score: int = Field(DEFAULT_VOLUME_SCORE, ge=0, le=100)
    progression_trend: float = Field(DEFAULT_PROGRESSION_TREND, ge=0, le=MAX_PROGRESSION_TREND)

    @field_validator("rir")
    @classmethod
    def _failure_is_zero_rir(cls, v: int) -> int:
        return max(v, 0)

    def as_row(self) -> dict[str, float | int | str]:
        return {col: getattr(self, col) for col in FEATURE_COLUMNS}


def to_frame(inputs: Sequence[ModelInput]) -> pd.DataFrame:
    """Build the model's input frame with columns in training order."""
    return pd.DataFrame([i.as_row() for i in inputs], columns=FEATURE_COLUMNS)


def set_intensity(rir: int) -> float:
    """0 (4+ RIR, easy) .. 1 (failure)."""
    return (5 - max(0, min(rir, 5))) / 5


def calculate_volume_score(rirs: Sequence[int], frequency: int, exercise: str) -> int:
    """Weekly training-stress score in [0, 100]."""
    total_intensity = sum(set_intensity(r) for r in rirs)
    raw = total_intensity * frequency * MUSCLE_MULTIPLIER.get(exercise, 1.0) * 8
    return int(np.clip(raw, 0, 100))


@dataclass(frozen=True)
class SessionSummary:
    """One logged workout reduced to what the features need."""

    performed_at: datetime
    top_weight: float
    top_reps: int
    top_rir: int
    set_rirs: tuple[int, ...]


@dataclass(frozen=True)
class TrainingContext:
    """History-derived features for one lifter and exercise."""

    last_weight: float
    last_reps: int
    last_rir: int
    num_sets: int
    frequency: int
    volume_score: int
    progression_trend: float


def compute_frequency(sessions: Sequence[SessionSummary]) -> int:
    """Sessions in the 7 days ending at the most recent session."""
    latest = sessions[-1].performed_at
    window_start = latest - timedelta(days=7)
    count = sum(1 for s in sessions if s.performed_at > window_start)
    return int(np.clip(count, 1, 14))


def compute_progression_trend(top_weights: Sequence[float]) -> float:
    """
    Least-squares slope of the top-set weight over the last five sessions,
    expressed in 5 lb increments per session and clipped to the training range.
    """
    recent = list(top_weights[-5:])
    if len(recent) < 2:
        return DEFAULT_PROGRESSION_TREND
    x = np.arange(len(recent), dtype=float)
    slope = float(np.polyfit(x, np.asarray(recent, dtype=float), 1)[0])
    return round(float(np.clip(slope / 5.0, 0.0, MAX_PROGRESSION_TREND)), 3)


def compute_context(sessions: Sequence[SessionSummary], exercise: str) -> TrainingContext | None:
    """Derive model features from sessions ordered oldest to newest."""
    if not sessions:
        return None
    latest = sessions[-1]
    frequency = compute_frequency(sessions)
    recent = sessions[-3:]
    num_sets = int(np.clip(round(sum(len(s.set_rirs) for s in recent) / len(recent)), 1, 20))
    return TrainingContext(
        last_weight=latest.top_weight,
        last_reps=latest.top_reps,
        last_rir=max(latest.top_rir, 0),
        num_sets=num_sets,
        frequency=frequency,
        volume_score=calculate_volume_score(latest.set_rirs, frequency, exercise),
        progression_trend=compute_progression_trend([s.top_weight for s in sessions]),
    )
