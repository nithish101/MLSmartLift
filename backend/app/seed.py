"""
Populate the database with ~4 weeks of demo workouts across all 3 exercises.

    python -m app.seed            # wipes existing workouts first
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Prediction, Workout, WorkoutSet, init_db, make_engine, make_session_factory
from app.features import ModelInput
from app.ml.predictor import RepModel, round_reps
from app.model_service import ModelService

DEMO_BODYWEIGHT = 180.0

# (weight, top-set reps, top-set rir, days ago)
DEMO_WORKOUTS: dict[str, list[tuple[float, int, int, int]]] = {
    "bench_press": [
        (135, 8, 3, 28), (135, 9, 2, 25), (140, 7, 2, 21), (140, 8, 2, 18), (145, 6, 2, 14),
        (145, 7, 2, 11), (145, 8, 1, 7), (150, 6, 2, 4), (150, 7, 2, 1),
    ],
    "squat": [
        (185, 8, 3, 27), (185, 9, 2, 24), (195, 7, 2, 20), (195, 8, 2, 17), (200, 6, 2, 13),
        (200, 7, 1, 10), (205, 6, 2, 6), (205, 7, 2, 3),
    ],
    "deadlift": [
        (225, 7, 3, 26), (225, 8, 2, 22), (235, 6, 2, 19), (235, 7, 2, 15), (245, 5, 2, 12),
        (245, 6, 1, 8), (250, 5, 2, 5), (250, 6, 2, 2),
    ],
}


def seed(db: Session, model: RepModel | None, now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    db.execute(delete(Prediction))
    db.execute(delete(WorkoutSet))
    db.execute(delete(Workout))

    total = 0
    for exercise, workouts in DEMO_WORKOUTS.items():
        for weight, reps, rir, days_ago in workouts:
            performed_at = now - timedelta(days=days_ago)
            workout = Workout(exercise=exercise, bodyweight=DEMO_BODYWEIGHT, created_at=performed_at)
            # Four sets; later sets lose a rep and gain RIR every two sets (fatigue).
            workout.sets = [
                WorkoutSet(
                    set_number=n,
                    weight=weight,
                    reps=max(1, reps - (n - 1) // 2),
                    rir=min(4, rir + (n - 1) // 2),
                )
                for n in range(1, 5)
            ]
            db.add(workout)
            db.flush()
            if model is not None:
                features = ModelInput(exercise=exercise, weight=weight, rir=rir, bodyweight=DEMO_BODYWEIGHT)  # type: ignore[arg-type]
                db.add(
                    Prediction(
                        workout_id=workout.id,
                        exercise=exercise,
                        predicted_reps=round_reps(model.predict([features])[0]),
                        actual_reps=reps,
                        weight=weight,
                        created_at=performed_at,
                    )
                )
            total += 1
    db.commit()
    return total


def main() -> None:  # pragma: no cover - CLI wrapper
    settings = get_settings()
    engine = make_engine(settings.database_url)
    init_db(engine)
    service = ModelService(settings.model_path)
    service.load()
    with make_session_factory(engine)() as db:
        total = seed(db, service if service.loaded else None)
    print(f"Seeded {total} demo workouts across {len(DEMO_WORKOUTS)} exercises")


if __name__ == "__main__":
    main()
