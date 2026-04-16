"""
Synthetic strength training data generator for SmartLift ML model.
Generates realistic progressive overload patterns with session-recoverable fatigue.
No deload weeks — fatigue is fully recovered between sessions.
"""

import numpy as np
import pandas as pd
import os

EXERCISES = {
    "bench_press": {"start_weight": (65, 135), "max_weight": (135, 315)},
    "squat": {"start_weight": (95, 185), "max_weight": (185, 405)},
    "deadlift": {"start_weight": (135, 225), "max_weight": (225, 500)},
}

RIR_VALUES = [0, 1, 2, 3, 4]


def generate_lifter_progression(lifter_id: int, exercise: str, num_sessions: int = 40):
    """Generate a realistic progression timeline for one lifter on one exercise.
    Fatigue from each session is fully recoverable before the next session."""
    rng = np.random.default_rng(lifter_id * 100 + hash(exercise) % 1000)

    config = EXERCISES[exercise]
    start_w = rng.uniform(*config["start_weight"])
    max_w = rng.uniform(*config["max_weight"])
    bodyweight = rng.uniform(140, 240)

    # Progression rate (lbs per session)
    progression_rate = rng.uniform(1.5, 5.0)

    records = []
    current_weight = start_w

    for session in range(num_sessions):
        # Session-level fatigue (fully recovered by next session)
        session_fatigue = rng.uniform(0.0, 1.5)

        session_weight = current_weight
        rir = rng.choice(RIR_VALUES, p=[0.05, 0.15, 0.35, 0.30, 0.15])

        # Round weight to nearest 5
        session_weight = round(session_weight / 5) * 5
        session_weight = max(session_weight, 45)

        # Calculate reps based on weight relative to max and session fatigue
        strength_ratio = session_weight / max_w
        base_reps = max(1, int(round(12 - 10 * strength_ratio)))
        fatigue_penalty = int(session_fatigue * 0.3)

        reps = base_reps - fatigue_penalty + rng.integers(-1, 2)
        reps = int(np.clip(reps, 1, 15))

        # Number of sets this session
        num_sets = rng.choice([3, 4, 5], p=[0.3, 0.5, 0.2])

        # Recent training frequency (sessions per week)
        frequency = rng.choice([2, 3, 4], p=[0.3, 0.5, 0.2])

        # Volume score estimate
        volume_score = min(100, int(num_sets * (5 - rir) * frequency * rng.uniform(2.0, 3.5)))

        # Progression trend: positive since we're always progressing
        trend = rng.uniform(0.0, 1.5)

        # Bodyweight fluctuation
        bw = bodyweight + rng.normal(0, 2)

        # Recovery time estimate (hours) — based on intensity and volume
        intensity_factor = (5 - rir) / 5  # 0 to 1
        muscle_recovery_base = {"bench_press": 36, "squat": 48, "deadlift": 48}
        base_recovery = muscle_recovery_base.get(exercise, 40)
        recovery_hours = base_recovery + (num_sets * intensity_factor * 4) + rng.normal(0, 3)
        recovery_hours = max(24, min(72, recovery_hours))

        records.append({
            "lifter_id": lifter_id,
            "session": session,
            "exercise": exercise,
            "weight": float(session_weight),
            "reps": int(reps),
            "rir": int(rir),
            "num_sets": int(num_sets),
            "bodyweight": round(float(bw), 1),
            "frequency": int(frequency),
            "volume_score": int(np.clip(volume_score, 0, 100)),
            "progression_trend": round(float(trend), 3),
            "recovery_hours": round(float(recovery_hours), 1),
        })

        # Always progress if reps are in range (no deloads)
        if reps >= 5:
            current_weight = min(current_weight + progression_rate * rng.uniform(0.5, 1.2), max_w)

    return records


def generate_dataset(num_lifters: int = 50, output_path: str = None):
    """Generate full synthetic dataset."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "training_data.csv")

    all_records = []
    for lifter_id in range(num_lifters):
        for exercise in EXERCISES:
            sessions = np.random.randint(25, 50)
            records = generate_lifter_progression(lifter_id, exercise, sessions)
            all_records.extend(records)

    df = pd.DataFrame(all_records)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} training samples → {output_path}")
    return df


if __name__ == "__main__":
    generate_dataset()
