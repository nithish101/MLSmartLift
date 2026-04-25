"""
SmartLift ML Predictor: inference, 1RM estimation, volume scoring, recovery estimation.
"""

import os
import joblib
import numpy as np

MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "smartlift_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "exercise_encoder.pkl")

EXERCISE_MAP = {"bench_press": 0, "deadlift": 1, "squat": 2}

# Base recovery hours by muscle group
MUSCLE_RECOVERY_BASE = {
    "bench_press": 36,
    "squat": 48,
    "deadlift": 48,
}


class SmartLiftPredictor:
    def __init__(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
        else:
            self.model = None
            self.encoder = None
            print("WARNING: Model not found. Run train_model.py first.")

    def _encode_exercise(self, exercise: str) -> int:
        if self.encoder is not None:
            try:
                return int(self.encoder.transform([exercise])[0])
            except ValueError:
                pass
        return EXERCISE_MAP.get(exercise, 0)

    def predict_reps(
        self,
        exercise: str,
        weight: float,
        rir: int,
        bodyweight: float,
        num_sets: int = 4,
        frequency: int = 3,
        volume_score: int = 50,
        progression_trend: float = 0.5,
    ) -> float:
        """Predict reps for a given weight and context."""
        if self.model is None:
            # Fallback heuristic
            return max(1, round(10 - (weight / bodyweight) * 5 + rir))

        features = np.array([[
            self._encode_exercise(exercise),
            weight,
            rir,
            bodyweight,
            num_sets,
            frequency,
            volume_score,
            progression_trend,
        ]])
        pred = self.model.predict(features)[0]
        return max(1, round(pred))

    def predict_next_session(
        self,
        exercise: str,
        current_weight: float,
        last_reps: int,
        last_rir: int,
        bodyweight: float,
        num_sets: int = 4,
        frequency: int = 3,
        volume_score: int = 50,
        progression_trend: float = 0.5,
    ) -> dict:
        """Generate full next-session recommendation."""
        # Predict reps at current weight with target RIR of 2
        target_rir = 2
        predicted_reps = self.predict_reps(
            exercise, current_weight, target_rir, bodyweight,
            num_sets, frequency, volume_score, progression_trend
        )

        # Determine weight adjustment
        recommended_weight = current_weight
        if predicted_reps > 9:
            recommended_weight = current_weight + 5
            predicted_reps = self.predict_reps(
                exercise, recommended_weight, target_rir, bodyweight,
                num_sets, frequency, volume_score, progression_trend
            )
        elif predicted_reps < 5:
            recommended_weight = max(45, current_weight - 5)
            predicted_reps = self.predict_reps(
                exercise, recommended_weight, target_rir, bodyweight,
                num_sets, frequency, volume_score, progression_trend
            )

        # Confidence score
        confidence = 1.0 - abs(predicted_reps - 7) / 7
        confidence = round(max(0.3, min(1.0, confidence)), 2)

        return {
            "exercise": exercise,
            "recommended_weight": round(recommended_weight / 5) * 5,
            "expected_reps": int(predicted_reps),
            "target_rir": target_rir,
            "confidence": confidence,
        }

    @staticmethod
    def estimate_1rm(weight: float, reps: int, rir: int = 0) -> float:
        """Estimate 1RM using modified Epley formula with RIR adjustment."""
        effective_reps = reps + rir
        if effective_reps <= 1:
            return weight
        e1rm = weight * (1 + effective_reps / 30)
        return round(e1rm, 1)

    @staticmethod
    def estimate_recovery_time(
        sets_data: list[dict],
        exercise: str = "bench_press",
    ) -> dict:
        """
        Estimate recovery time (hours) after a workout.
        Fatigue is fully recoverable before the next session.

        Returns dict with recovery_hours and a human-readable estimate.
        """
        base_hours = MUSCLE_RECOVERY_BASE.get(exercise, 40)

        total_intensity = 0
        for s in sets_data:
            rir = s.get("rir", 2)
            intensity = (5 - rir) / 5  # 0 to 1
            total_intensity += intensity

        # More intense sets add more recovery time
        additional_hours = total_intensity * 4
        recovery_hours = base_hours + additional_hours
        recovery_hours = max(24, min(72, recovery_hours))

        # Human-readable format
        days = int(recovery_hours // 24)
        remaining_hours = int(recovery_hours % 24)
        if days > 0 and remaining_hours > 0:
            readable = f"{days} day{'s' if days > 1 else ''} {remaining_hours} hours"
        elif days > 0:
            readable = f"{days} day{'s' if days > 1 else ''}"
        else:
            readable = f"{int(recovery_hours)} hours"

        return {
            "recovery_hours": round(recovery_hours, 1),
            "recovery_estimate": readable,
        }

    @staticmethod
    def calculate_volume_score(
        sets_data: list[dict],
        frequency: int = 3,
        exercise: str = "bench_press",
    ) -> int:
        """
        Calculate weekly training stress score (0-100).
        """
        muscle_multiplier = {
            "squat": 1.3,
            "deadlift": 1.4,
            "bench_press": 1.0,
        }
        multiplier = muscle_multiplier.get(exercise, 1.0)

        total_intensity = 0
        for s in sets_data:
            rir = s.get("rir", 2)
            intensity = (5 - rir) / 5
            total_intensity += intensity

        raw_score = total_intensity * frequency * multiplier * 8
        return int(np.clip(raw_score, 0, 100))

    def generate_alerts(
        self,
        recent_workouts: list[dict],
        volume_score: int,
        exercise: str,
    ) -> list[dict]:
        """Generate progression alerts based on recent performance."""
        alerts = []

        if len(recent_workouts) < 2:
            return alerts

        # Check progression trend
        weights = [w.get("weight", 0) for w in recent_workouts[-5:]]
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

        # Volume alerts
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

        # RIR-based suggestions
        rirs = [w.get("rir", 2) for w in recent_workouts[-3:]]
        avg_rir = sum(rirs) / len(rirs) if rirs else 2
        if avg_rir > 3:
            alerts.append({
                "type": "tip",
                "message": "Average RIR is high. You could increase intensity for better stimulus.",
            })

        return alerts
