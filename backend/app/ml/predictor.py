"""
SmartLift domain logic: next-session recommendation, 1RM estimation,
recovery estimation and progression alerts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from app.features import ModelInput, TrainingContext, set_intensity

# Base recovery hours by muscle group
MUSCLE_RECOVERY_BASE: dict[str, float] = {
    "bench_press": 36,
    "squat": 48,
    "deadlift": 48,
}

# Rep window the recommendation engine steers toward (at TARGET_RIR).
TARGET_RIR = 2
MIN_TARGET_REPS = 5
MAX_TARGET_REPS = 9
WEIGHT_STEP = 5.0
MIN_BAR_WEIGHT = 45.0

DEFAULT_START_WEIGHT: dict[str, float] = {"bench_press": 95, "squat": 135, "deadlift": 135}

Outcome = Literal["decrease", "maintain", "increase"]


class RepModel(Protocol):
    def predict(self, inputs: Sequence[ModelInput]) -> list[float]: ...


def round_reps(pred: float) -> int:
    return max(1, round(pred))


def outcome_for(predicted_reps: int) -> Outcome:
    """Bucket a prediction by the weight change it implies."""
    if predicted_reps > MAX_TARGET_REPS:
        return "increase"
    if predicted_reps < MIN_TARGET_REPS:
        return "decrease"
    return "maintain"


def recommend_next_session(
    model: RepModel,
    exercise: str,
    bodyweight: float,
    context: TrainingContext,
) -> dict[str, float | int | str]:
    """Pick next session's load so predicted reps at RIR 2 land in the 5–9 window."""

    def predict_at(weight: float) -> int:
        features = ModelInput(
            exercise=exercise,  # type: ignore[arg-type]
            weight=weight,
            rir=TARGET_RIR,
            bodyweight=bodyweight,
            num_sets=context.num_sets,
            frequency=context.frequency,
            volume_score=context.volume_score,
            progression_trend=context.progression_trend,
        )
        return round_reps(model.predict([features])[0])

    recommended_weight = context.last_weight
    predicted_reps = predict_at(recommended_weight)
    outcome = outcome_for(predicted_reps)
    if outcome == "increase":
        recommended_weight = context.last_weight + WEIGHT_STEP
        predicted_reps = predict_at(recommended_weight)
    elif outcome == "decrease":
        recommended_weight = max(MIN_BAR_WEIGHT, context.last_weight - WEIGHT_STEP)
        predicted_reps = predict_at(recommended_weight)

    # Highest confidence when the prediction sits mid-window (7 reps).
    confidence = 1.0 - abs(predicted_reps - 7) / 7
    confidence = round(max(0.3, min(1.0, confidence)), 2)

    return {
        "exercise": exercise,
        "recommended_weight": round(recommended_weight / WEIGHT_STEP) * WEIGHT_STEP,
        "expected_reps": int(predicted_reps),
        "target_rir": TARGET_RIR,
        "confidence": confidence,
    }


def estimate_1rm(weight: float, reps: int, rir: int = 0) -> float:
    """Estimate 1RM using the Epley formula, counting reps in reserve as reps."""
    effective_reps = reps + max(rir, 0)
    if effective_reps <= 1:
        return weight
    return round(weight * (1 + effective_reps / 30), 1)


def estimate_recovery_time(set_rirs: Sequence[int], exercise: str = "bench_press") -> dict[str, float | str]:
    """Estimate hours to recover after a session, clamped to 24–72h."""
    base_hours = MUSCLE_RECOVERY_BASE.get(exercise, 40)
    total_intensity = sum(set_intensity(r) for r in set_rirs)
    recovery_hours = max(24.0, min(72.0, base_hours + total_intensity * 4))

    days = int(recovery_hours // 24)
    remaining_hours = int(recovery_hours % 24)
    if days > 0 and remaining_hours > 0:
        readable = f"{days} day{'s' if days > 1 else ''} {remaining_hours} hours"
    else:
        readable = f"{days} day{'s' if days > 1 else ''}"

    return {"recovery_hours": round(recovery_hours, 1), "recovery_estimate": readable}


def generate_alerts(
    recent_weights: Sequence[float],
    recent_rirs: Sequence[int],
    volume_score: int,
) -> list[dict[str, str]]:
    """Progression alerts from recent top-set weights and RIRs (oldest to newest)."""
    alerts: list[dict[str, str]] = []
    if len(recent_weights) < 2:
        return alerts

    weights = list(recent_weights[-5:])
    if len(weights) >= 3:
        trend = weights[-1] - weights[0]
        if trend < 0:
            alerts.append({
                "type": "warning",
                "message": "Progress slower than expected. Consider adjusting volume or recovery.",
            })
        elif trend == 0 and len(weights) >= 4:
            alerts.append({
                "type": "info",
                "message": "Weight has plateaued. Try adding sets or increasing frequency.",
            })

    if volume_score > 80:
        alerts.append({
            "type": "warning",
            "message": "High training stress. Make sure you're recovering fully between sessions.",
        })
    elif volume_score < 25:
        alerts.append({
            "type": "info",
            "message": "Training volume may be too low. Consider adding sets or sessions.",
        })

    rirs = list(recent_rirs[-3:])
    if rirs and sum(rirs) / len(rirs) > 3:
        alerts.append({
            "type": "tip",
            "message": "Average RIR is high. You could increase intensity for better stimulus.",
        })

    return alerts
