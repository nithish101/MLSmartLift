"""
Pydantic models for SmartLift API.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# --- Request Models ---

class SetLog(BaseModel):
    weight: float = Field(..., gt=0, description="Weight in lbs")
    reps: int = Field(..., ge=1, le=30, description="Reps completed")
    rir: int = Field(..., ge=-1, le=4, description="Reps in reserve (-1 = failure)")


class WorkoutCreate(BaseModel):
    exercise: str = Field(..., description="Exercise name: bench_press, squat, deadlift")
    bodyweight: float = Field(..., gt=0, description="User bodyweight in lbs")
    sets: list[SetLog] = Field(..., min_length=1, description="List of sets performed")


# --- Response Models ---

class SetResponse(BaseModel):
    set_number: int
    weight: float
    reps: int
    rir: int


class RecoveryEstimate(BaseModel):
    recovery_hours: float
    recovery_estimate: str


class WorkoutResponse(BaseModel):
    id: int
    exercise: str
    bodyweight: float
    sets: list[SetResponse]
    recovery: RecoveryEstimate
    created_at: str


class Recommendation(BaseModel):
    exercise: str
    recommended_weight: float
    expected_reps: int
    target_rir: int
    confidence: float


class ProgressPoint(BaseModel):
    date: str
    estimated_1rm: float
    weight: float
    reps: int
    rir: int
    exercise: str


class VolumeScoreResponse(BaseModel):
    current_score: int
    interpretation: str
    weekly_scores: list[dict]


class PredictedVsActual(BaseModel):
    date: str
    predicted_reps: int
    actual_reps: int
    weight: float


class Alert(BaseModel):
    type: str
    message: str
