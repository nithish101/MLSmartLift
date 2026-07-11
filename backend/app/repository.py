"""Database reads used by the API and feature computation."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Prediction, Workout
from app.features import SessionSummary


def recent_workouts(db: Session, exercise: str | None, limit: int) -> list[Workout]:
    """Most recent workouts first."""
    stmt = select(Workout).order_by(Workout.created_at.desc(), Workout.id.desc()).limit(limit)
    if exercise:
        stmt = stmt.where(Workout.exercise == exercise)
    return list(db.scalars(stmt))


def all_workouts(db: Session, exercise: str) -> list[Workout]:
    """All workouts for an exercise, oldest first."""
    stmt = (
        select(Workout)
        .where(Workout.exercise == exercise)
        .order_by(Workout.created_at.asc(), Workout.id.asc())
    )
    return list(db.scalars(stmt))


def summarize(workout: Workout) -> SessionSummary:
    top = max(workout.sets, key=lambda s: (s.weight, s.reps))
    return SessionSummary(
        performed_at=workout.created_at,
        top_weight=top.weight,
        top_reps=top.reps,
        top_rir=top.rir,
        set_rirs=tuple(s.rir for s in workout.sets),
    )


def session_history(db: Session, exercise: str, limit: int = 20) -> list[SessionSummary]:
    """Summaries of the last ``limit`` sessions, oldest first."""
    workouts = recent_workouts(db, exercise, limit)
    return [summarize(w) for w in reversed(workouts) if w.sets]


def predictions_for(db: Session, exercise: str) -> list[Prediction]:
    stmt = (
        select(Prediction)
        .where(Prediction.exercise == exercise)
        .order_by(Prediction.created_at.asc(), Prediction.id.asc())
    )
    return list(db.scalars(stmt))
