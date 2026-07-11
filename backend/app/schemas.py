"""Request/response schemas for the SmartLift API."""

from pydantic import BaseModel, Field

from app.features import Exercise, ModelInput

# --- Requests ---


class SetLog(BaseModel):
    weight: float = Field(..., gt=0, le=1500, description="Weight in lbs")
    reps: int = Field(..., ge=1, le=30, description="Reps completed")
    rir: int = Field(..., ge=-1, le=4, description="Reps in reserve (-1 = failure)")


class WorkoutCreate(BaseModel):
    exercise: Exercise
    bodyweight: float = Field(..., ge=50, le=700, description="User bodyweight in lbs")
    sets: list[SetLog] = Field(..., min_length=1, max_length=20)


class PredictRequest(ModelInput):
    """Feature vector for a single rep prediction."""


# --- Responses ---


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


class PredictResponse(BaseModel):
    predicted_reps: int
    raw_prediction: float
    outcome: str


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


class WeeklyScore(BaseModel):
    week: str
    score: int


class VolumeScoreResponse(BaseModel):
    current_score: int
    interpretation: str
    weekly_scores: list[WeeklyScore]


class PredictedVsActual(BaseModel):
    date: str
    predicted_reps: int
    actual_reps: int
    weight: float


class Alert(BaseModel):
    type: str
    message: str


class HealthResponse(BaseModel):
    status: str
    database: str
    model_loaded: bool
