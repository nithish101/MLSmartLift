"""
SmartLift FastAPI Backend
AI-powered strength training assistant API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from database import get_connection, init_db
from models import (
    WorkoutCreate, WorkoutResponse, SetResponse, RecoveryEstimate,
    Recommendation, ProgressPoint, VolumeScoreResponse,
    PredictedVsActual, Alert,
)
from ml.predictor import SmartLiftPredictor


predictor = SmartLiftPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="SmartLift API",
    description="AI-powered strength training assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/workouts", response_model=WorkoutResponse)
async def create_workout(workout: WorkoutCreate):
    """Log a new workout with sets."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO workouts (exercise, bodyweight) VALUES (?, ?)",
            (workout.exercise, workout.bodyweight),
        )
        workout_id = cursor.lastrowid

        set_responses = []
        sets_for_recovery = []
        for i, s in enumerate(workout.sets, 1):
            rir_val = s.rir if s.rir >= 0 else 0  # -1 (failure) maps to 0
            cursor.execute(
                "INSERT INTO sets (workout_id, set_number, weight, reps, rir) VALUES (?, ?, ?, ?, ?)",
                (workout_id, i, s.weight, s.reps, rir_val),
            )
            set_responses.append(SetResponse(
                set_number=i, weight=s.weight, reps=s.reps, rir=rir_val
            ))
            sets_for_recovery.append({"weight": s.weight, "reps": s.reps, "rir": rir_val})

        # Store prediction for predicted vs actual tracking
        top_set = max(workout.sets, key=lambda s: s.weight)
        predicted_reps = predictor.predict_reps(
            workout.exercise, top_set.weight, top_set.rir if top_set.rir >= 0 else 0,
            workout.bodyweight,
        )
        cursor.execute(
            "INSERT INTO predictions (workout_id, exercise, predicted_reps, actual_reps, weight) VALUES (?, ?, ?, ?, ?)",
            (workout_id, workout.exercise, int(predicted_reps), top_set.reps, top_set.weight),
        )

        conn.commit()

        # Calculate recovery estimate
        recovery = predictor.estimate_recovery_time(sets_for_recovery, workout.exercise)

        return WorkoutResponse(
            id=workout_id,
            exercise=workout.exercise,
            bodyweight=workout.bodyweight,
            sets=set_responses,
            recovery=RecoveryEstimate(**recovery),
            created_at=datetime.now().isoformat(),
        )
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/workouts", response_model=list[WorkoutResponse])
async def get_workouts(exercise: str = None, limit: int = 50):
    """Get workout history."""
    conn = get_connection()
    try:
        if exercise:
            workouts = conn.execute(
                "SELECT * FROM workouts WHERE exercise = ? ORDER BY created_at DESC LIMIT ?",
                (exercise, limit),
            ).fetchall()
        else:
            workouts = conn.execute(
                "SELECT * FROM workouts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        results = []
        for w in workouts:
            sets = conn.execute(
                "SELECT * FROM sets WHERE workout_id = ? ORDER BY set_number",
                (w["id"],),
            ).fetchall()
            sets_list = [SetResponse(
                set_number=s["set_number"],
                weight=s["weight"],
                reps=s["reps"],
                rir=s["rir"],
            ) for s in sets]

            recovery = predictor.estimate_recovery_time(
                [dict(s) for s in sets], w["exercise"]
            )

            results.append(WorkoutResponse(
                id=w["id"],
                exercise=w["exercise"],
                bodyweight=w["bodyweight"],
                sets=sets_list,
                recovery=RecoveryEstimate(**recovery),
                created_at=w["created_at"],
            ))
        return results
    finally:
        conn.close()


@app.get("/recommendation/{exercise}", response_model=Recommendation)
async def get_recommendation(exercise: str, bodyweight: float = 180):
    """Get AI recommendation for next workout."""
    conn = get_connection()
    try:
        latest = conn.execute(
            "SELECT w.*, s.weight, s.reps, s.rir FROM workouts w "
            "JOIN sets s ON w.id = s.workout_id "
            "WHERE w.exercise = ? ORDER BY w.created_at DESC, s.weight DESC LIMIT 1",
            (exercise,),
        ).fetchone()

        if not latest:
            defaults = {"bench_press": 95, "squat": 135, "deadlift": 135}
            return Recommendation(
                exercise=exercise,
                recommended_weight=defaults.get(exercise, 95),
                expected_reps=8,
                target_rir=2,
                confidence=0.5,
            )

        recent_sets = conn.execute(
            "SELECT s.* FROM sets s JOIN workouts w ON s.workout_id = w.id "
            "WHERE w.exercise = ? ORDER BY w.created_at DESC LIMIT 20",
            (exercise,),
        ).fetchall()

        volume_score = predictor.calculate_volume_score(
            [dict(s) for s in recent_sets[-8:]], frequency=3, exercise=exercise,
        )

        rec = predictor.predict_next_session(
            exercise=exercise,
            current_weight=latest["weight"],
            last_reps=latest["reps"],
            last_rir=latest["rir"],
            bodyweight=bodyweight,
            volume_score=volume_score,
        )

        return Recommendation(**rec)
    finally:
        conn.close()


@app.get("/progress/{exercise}", response_model=list[ProgressPoint])
async def get_progress(exercise: str):
    """Get estimated 1RM progress over time."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT w.created_at, w.exercise, s.weight, s.reps, s.rir "
            "FROM workouts w "
            "JOIN sets s ON w.id = s.workout_id "
            "WHERE w.exercise = ? "
            "ORDER BY w.created_at ASC",
            (exercise,),
        ).fetchall()

        seen_dates = {}
        for r in rows:
            date = r["created_at"][:10] if r["created_at"] else datetime.now().strftime("%Y-%m-%d")
            e1rm = predictor.estimate_1rm(r["weight"], r["reps"], r["rir"])

            if date not in seen_dates or e1rm > seen_dates[date]["estimated_1rm"]:
                seen_dates[date] = {
                    "date": date,
                    "estimated_1rm": e1rm,
                    "weight": r["weight"],
                    "reps": r["reps"],
                    "rir": r["rir"],
                    "exercise": exercise,
                }

        progress = [ProgressPoint(**v) for v in seen_dates.values()]
        return sorted(progress, key=lambda p: p.date)
    finally:
        conn.close()


@app.get("/volume-score", response_model=VolumeScoreResponse)
async def get_volume_score(exercise: str = "bench_press"):
    """Get current volume score and weekly history."""
    conn = get_connection()
    try:
        recent_sets = conn.execute(
            "SELECT s.* FROM sets s JOIN workouts w ON s.workout_id = w.id "
            "WHERE w.exercise = ? ORDER BY w.created_at DESC LIMIT 20",
            (exercise,),
        ).fetchall()

        current = predictor.calculate_volume_score(
            [dict(s) for s in recent_sets], frequency=3, exercise=exercise,
        )

        if current < 30:
            interp = "Low volume — likely undertraining. Consider adding sets or sessions."
        elif current < 60:
            interp = "Moderate volume — productive training load. Keep it up!"
        elif current < 80:
            interp = "High volume — effective but watch for fatigue accumulation."
        else:
            interp = "Very high volume — potential overreaching. Consider reducing intensity."

        all_workouts = conn.execute(
            "SELECT w.created_at, s.weight, s.reps, s.rir "
            "FROM workouts w JOIN sets s ON w.id = s.workout_id "
            "WHERE w.exercise = ? ORDER BY w.created_at ASC",
            (exercise,),
        ).fetchall()

        weekly_scores = []
        if all_workouts:
            weeks = {}
            for w in all_workouts:
                date_str = w["created_at"][:10] if w["created_at"] else datetime.now().strftime("%Y-%m-%d")
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    dt = datetime.now()
                week_key = dt.strftime("%Y-W%U")
                if week_key not in weeks:
                    weeks[week_key] = []
                weeks[week_key].append(dict(w))

            for week, sets in sorted(weeks.items()):
                score = predictor.calculate_volume_score(sets, 3, exercise)
                weekly_scores.append({"week": week, "score": score})

        return VolumeScoreResponse(
            current_score=current,
            interpretation=interp,
            weekly_scores=weekly_scores[-8:],
        )
    finally:
        conn.close()


@app.get("/predicted-vs-actual/{exercise}", response_model=list[PredictedVsActual])
async def get_predicted_vs_actual(exercise: str):
    """Get predicted vs actual rep comparison."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT p.created_at, p.predicted_reps, p.actual_reps, p.weight "
            "FROM predictions p WHERE p.exercise = ? ORDER BY p.created_at ASC",
            (exercise,),
        ).fetchall()

        return [PredictedVsActual(
            date=r["created_at"][:10] if r["created_at"] else datetime.now().strftime("%Y-%m-%d"),
            predicted_reps=r["predicted_reps"],
            actual_reps=r["actual_reps"],
            weight=r["weight"],
        ) for r in rows]
    finally:
        conn.close()


@app.get("/alerts/{exercise}", response_model=list[Alert])
async def get_alerts(exercise: str):
    """Get progression alerts."""
    conn = get_connection()
    try:
        recent = conn.execute(
            "SELECT s.weight, s.reps, s.rir FROM sets s "
            "JOIN workouts w ON s.workout_id = w.id "
            "WHERE w.exercise = ? ORDER BY w.created_at DESC LIMIT 15",
            (exercise,),
        ).fetchall()

        recent_sets = conn.execute(
            "SELECT s.* FROM sets s JOIN workouts w ON s.workout_id = w.id "
            "WHERE w.exercise = ? ORDER BY w.created_at DESC LIMIT 20",
            (exercise,),
        ).fetchall()

        volume_score = predictor.calculate_volume_score(
            [dict(s) for s in recent_sets], 3, exercise
        )

        alerts = predictor.generate_alerts(
            [dict(r) for r in recent], volume_score, exercise,
        )

        return [Alert(**a) for a in alerts]
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
