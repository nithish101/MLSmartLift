"""
Seed script to populate the SmartLift database with demo workout data.
Creates ~4 weeks of realistic training history across all 3 exercises.
"""

import sqlite3
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from database import init_db, DB_PATH
from ml.predictor import SmartLiftPredictor

predictor = SmartLiftPredictor()

DEMO_BODYWEIGHT = 180

# Realistic 4-week progression for each exercise
DEMO_WORKOUTS = {
    "bench_press": [
        {"weight": 135, "reps": 8, "rir": 3, "day_offset": -28},
        {"weight": 135, "reps": 9, "rir": 2, "day_offset": -25},
        {"weight": 140, "reps": 7, "rir": 2, "day_offset": -21},
        {"weight": 140, "reps": 8, "rir": 2, "day_offset": -18},
        {"weight": 145, "reps": 6, "rir": 2, "day_offset": -14},
        {"weight": 145, "reps": 7, "rir": 2, "day_offset": -11},
        {"weight": 145, "reps": 8, "rir": 1, "day_offset": -7},
        {"weight": 150, "reps": 6, "rir": 2, "day_offset": -4},
        {"weight": 150, "reps": 7, "rir": 2, "day_offset": -1},
    ],
    "squat": [
        {"weight": 185, "reps": 8, "rir": 3, "day_offset": -27},
        {"weight": 185, "reps": 9, "rir": 2, "day_offset": -24},
        {"weight": 195, "reps": 7, "rir": 2, "day_offset": -20},
        {"weight": 195, "reps": 8, "rir": 2, "day_offset": -17},
        {"weight": 200, "reps": 6, "rir": 2, "day_offset": -13},
        {"weight": 200, "reps": 7, "rir": 1, "day_offset": -10},
        {"weight": 205, "reps": 6, "rir": 2, "day_offset": -6},
        {"weight": 205, "reps": 7, "rir": 2, "day_offset": -3},
    ],
    "deadlift": [
        {"weight": 225, "reps": 7, "rir": 3, "day_offset": -26},
        {"weight": 225, "reps": 8, "rir": 2, "day_offset": -22},
        {"weight": 235, "reps": 6, "rir": 2, "day_offset": -19},
        {"weight": 235, "reps": 7, "rir": 2, "day_offset": -15},
        {"weight": 245, "reps": 5, "rir": 2, "day_offset": -12},
        {"weight": 245, "reps": 6, "rir": 1, "day_offset": -8},
        {"weight": 250, "reps": 5, "rir": 2, "day_offset": -5},
        {"weight": 250, "reps": 6, "rir": 2, "day_offset": -2},
    ],
}


def seed_database():
    """Populate database with demo data."""
    # Remove existing DB
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now()
    total_workouts = 0

    for exercise, workouts in DEMO_WORKOUTS.items():
        for w in workouts:
            workout_date = (now + timedelta(days=w["day_offset"])).strftime("%Y-%m-%d %H:%M:%S")

            # Insert workout
            cursor.execute(
                "INSERT INTO workouts (exercise, bodyweight, created_at) VALUES (?, ?, ?)",
                (exercise, DEMO_BODYWEIGHT, workout_date),
            )
            workout_id = cursor.lastrowid

            # Insert 4 sets with slight variation
            for set_num in range(1, 5):
                # Sets 2-4 have slightly fewer reps (fatigue within session)
                set_reps = max(1, w["reps"] - (set_num - 1) // 2)
                set_rir = min(4, w["rir"] + (set_num - 1) // 2)

                cursor.execute(
                    "INSERT INTO sets (workout_id, set_number, weight, reps, rir) VALUES (?, ?, ?, ?, ?)",
                    (workout_id, set_num, w["weight"], set_reps, set_rir),
                )

            # Store prediction
            predicted_reps = predictor.predict_reps(
                exercise, w["weight"], w["rir"], DEMO_BODYWEIGHT,
            )
            cursor.execute(
                "INSERT INTO predictions (workout_id, exercise, predicted_reps, actual_reps, weight, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (workout_id, exercise, int(predicted_reps), w["reps"], w["weight"], workout_date),
            )

            total_workouts += 1

    conn.commit()
    conn.close()
    print(f"✅ Seeded {total_workouts} demo workouts across 3 exercises")
    print(f"   Database: {DB_PATH}")


if __name__ == "__main__":
    seed_database()
