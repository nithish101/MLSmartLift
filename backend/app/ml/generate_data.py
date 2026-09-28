"""
Synthetic strength training data generator for the SmartLift model.
Generates realistic progressive overload patterns with session-recoverable fatigue.
No deload weeks: fatigue is fully recovered between sessions.
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXERCISES: dict[str, dict[str, tuple[float, float]]] = {
    "bench_press": {"start_weight": (65, 135), "max_weight": (135, 315)},
    "squat": {"start_weight": (95, 185), "max_weight": (185, 405)},
    "deadlift": {"start_weight": (135, 225), "max_weight": (225, 500)},
}

RIR_VALUES = [0, 1, 2, 3, 4]
MUSCLE_RECOVERY_BASE = {"bench_press": 36, "squat": 48, "deadlift": 48}


def _lifter_seed(seed: int, lifter_id: int, exercise: str) -> int:
    # crc32 rather than hash(): str hashes are randomized per process.
    return seed * 1_000_003 + lifter_id * 100 + zlib.crc32(exercise.encode()) % 1000


def generate_lifter_progression(
    lifter_id: int, exercise: str, num_sessions: int = 40, seed: int = 0
) -> list[dict[str, Any]]:
    """Progression timeline for one lifter on one exercise."""
    rng = np.random.default_rng(_lifter_seed(seed, lifter_id, exercise))

    config = EXERCISES[exercise]
    start_w = rng.uniform(*config["start_weight"])
    max_w = rng.uniform(*config["max_weight"])
    bodyweight = rng.uniform(140, 240)
    progression_rate = rng.uniform(1.5, 5.0)  # lbs per session

    records: list[dict[str, Any]] = []
    current_weight = start_w

    for session in range(num_sessions):
        session_fatigue = rng.uniform(0.0, 1.5)
        rir = int(rng.choice(RIR_VALUES, p=[0.05, 0.15, 0.35, 0.30, 0.15]))
        session_weight = max(round(current_weight / 5) * 5, 45)

        strength_ratio = session_weight / max_w
        base_reps = max(1, round(12 - 10 * strength_ratio))
        fatigue_penalty = int(session_fatigue * 0.3)
        reps = int(np.clip(base_reps - fatigue_penalty + rng.integers(-1, 2), 1, 15))

        num_sets = int(rng.choice([3, 4, 5], p=[0.3, 0.5, 0.2]))
        frequency = int(rng.choice([2, 3, 4], p=[0.3, 0.5, 0.2]))
        volume_score = min(100, int(num_sets * (5 - rir) * frequency * rng.uniform(2.0, 3.5)))
        trend = rng.uniform(0.0, 1.5)
        bw = bodyweight + rng.normal(0, 2)

        intensity_factor = (5 - rir) / 5
        base_recovery = MUSCLE_RECOVERY_BASE.get(exercise, 40)
        recovery_hours = base_recovery + (num_sets * intensity_factor * 4) + rng.normal(0, 3)
        recovery_hours = max(24, min(72, recovery_hours))

        records.append(
            {
                "lifter_id": lifter_id,
                "session": session,
                "exercise": exercise,
                "weight": float(session_weight),
                "reps": reps,
                "rir": rir,
                "num_sets": num_sets,
                "bodyweight": round(float(bw), 1),
                "frequency": frequency,
                "volume_score": int(np.clip(volume_score, 0, 100)),
                "progression_trend": round(float(trend), 3),
                "recovery_hours": round(float(recovery_hours), 1),
            }
        )

        if reps >= 5:
            current_weight = min(current_weight + progression_rate * rng.uniform(0.5, 1.2), max_w)

    return records


def generate_dataset(num_lifters: int = 50, seed: int = 0) -> pd.DataFrame:
    """Generate the full synthetic dataset deterministically from ``seed``."""
    rng = np.random.default_rng(seed)
    all_records: list[dict[str, Any]] = []
    for lifter_id in range(num_lifters):
        for exercise in EXERCISES:
            sessions = int(rng.integers(25, 50))
            all_records.extend(generate_lifter_progression(lifter_id, exercise, sessions, seed))
    return pd.DataFrame(all_records)


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/training_data.csv"))
    args = parser.parse_args()

    df = generate_dataset(args.lifters, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} training samples -> {args.out}")


if __name__ == "__main__":
    main()
