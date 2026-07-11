"""
SmartLift FastAPI backend.
AI-powered strength training assistant API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import repository
from app.config import Settings, get_settings
from app.db import Prediction, Workout, WorkoutSet, init_db, make_engine, make_session_factory
from app.features import Exercise, ModelInput, calculate_volume_score, compute_context
from app.ml import predictor
from app.model_service import ModelService, ModelUnavailableError
from app.schemas import (
    Alert,
    HealthResponse,
    PredictedVsActual,
    PredictRequest,
    PredictResponse,
    ProgressPoint,
    Recommendation,
    RecoveryEstimate,
    SetResponse,
    VolumeScoreResponse,
    WeeklyScore,
    WorkoutCreate,
    WorkoutResponse,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url, settings.db_pool_size, settings.db_max_overflow)
        init_db(engine)
        model_service = ModelService(settings.model_path)
        model_service.load()
        app.state.engine = engine
        app.state.session_factory = make_session_factory(engine)
        app.state.model_service = model_service
        yield
        engine.dispose()

    app = FastAPI(
        title=settings.app_name,
        description="AI-powered strength training assistant",
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_routes(app)
    return app


# ─── Dependencies ────────────────────────────────────────────────────────────


def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


def get_model(request: Request) -> ModelService:
    service: ModelService = request.app.state.model_service
    return service


DB = Annotated[Session, Depends(get_db)]
Model = Annotated[ModelService, Depends(get_model)]


def _serialize_workout(w: Workout) -> WorkoutResponse:
    rirs = [s.rir for s in w.sets]
    return WorkoutResponse(
        id=w.id,
        exercise=w.exercise,
        bodyweight=w.bodyweight,
        sets=[SetResponse(set_number=s.set_number, weight=s.weight, reps=s.reps, rir=s.rir) for s in w.sets],
        recovery=RecoveryEstimate(**predictor.estimate_recovery_time(rirs, w.exercise)),  # type: ignore[arg-type]
        created_at=w.created_at.isoformat(),
    )


def _model_unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail="Model not loaded")


# ─── Routes ──────────────────────────────────────────────────────────────────


def register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    def health(db: DB, model: Model) -> HealthResponse:
        try:
            db.execute(text("SELECT 1"))
            database = "ok"
        except Exception:
            database = "unavailable"
        status = "ok" if database == "ok" and model.loaded else "degraded"
        return HealthResponse(status=status, database=database, model_loaded=model.loaded)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, model: Model) -> PredictResponse:
        """Predict reps for one feature vector."""
        try:
            raw = model.predict([req])[0]
        except ModelUnavailableError as exc:
            raise _model_unavailable() from exc
        reps = predictor.round_reps(raw)
        return PredictResponse(predicted_reps=reps, raw_prediction=raw, outcome=predictor.outcome_for(reps))

    @app.post("/workouts", response_model=WorkoutResponse)
    def create_workout(workout: WorkoutCreate, db: DB, model: Model) -> WorkoutResponse:
        """Log a new workout with sets."""
        row = Workout(exercise=workout.exercise, bodyweight=workout.bodyweight)
        row.sets = [
            WorkoutSet(set_number=i, weight=s.weight, reps=s.reps, rir=max(s.rir, 0))
            for i, s in enumerate(workout.sets, 1)
        ]
        db.add(row)
        db.flush()

        # Store the model's prediction for the top set, for predicted-vs-actual tracking.
        if model.loaded:
            top = max(workout.sets, key=lambda s: s.weight)
            features = ModelInput(
                exercise=workout.exercise, weight=top.weight, rir=top.rir, bodyweight=workout.bodyweight
            )
            predicted = predictor.round_reps(model.predict([features])[0])
            db.add(
                Prediction(
                    workout_id=row.id,
                    exercise=workout.exercise,
                    predicted_reps=predicted,
                    actual_reps=top.reps,
                    weight=top.weight,
                )
            )
        db.commit()
        db.refresh(row)
        return _serialize_workout(row)

    @app.get("/workouts", response_model=list[WorkoutResponse])
    def get_workouts(
        db: DB,
        exercise: Exercise | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> list[WorkoutResponse]:
        """Workout history, newest first."""
        return [_serialize_workout(w) for w in repository.recent_workouts(db, exercise, limit)]

    @app.get("/recommendation/{exercise}", response_model=Recommendation)
    def get_recommendation(
        exercise: Exercise,
        db: DB,
        model: Model,
        bodyweight: Annotated[float, Query(ge=50, le=700)] = 180,
    ) -> Recommendation:
        """AI recommendation for the next session."""
        context = compute_context(repository.session_history(db, exercise), exercise)
        if context is None:
            return Recommendation(
                exercise=exercise,
                recommended_weight=predictor.DEFAULT_START_WEIGHT[exercise],
                expected_reps=8,
                target_rir=predictor.TARGET_RIR,
                confidence=0.5,
            )
        try:
            rec = predictor.recommend_next_session(model, exercise, bodyweight, context)
        except ModelUnavailableError as exc:
            raise _model_unavailable() from exc
        return Recommendation(**rec)  # type: ignore[arg-type]

    @app.get("/progress/{exercise}", response_model=list[ProgressPoint])
    def get_progress(exercise: Exercise, db: DB) -> list[ProgressPoint]:
        """Best estimated 1RM per day."""
        best: dict[str, ProgressPoint] = {}
        for w in repository.all_workouts(db, exercise):
            date = w.created_at.date().isoformat()
            for s in w.sets:
                e1rm = predictor.estimate_1rm(s.weight, s.reps, s.rir)
                if date not in best or e1rm > best[date].estimated_1rm:
                    best[date] = ProgressPoint(
                        date=date, estimated_1rm=e1rm, weight=s.weight, reps=s.reps, rir=s.rir, exercise=exercise
                    )
        return sorted(best.values(), key=lambda p: p.date)

    @app.get("/volume-score", response_model=VolumeScoreResponse)
    def get_volume_score(db: DB, exercise: Exercise = "bench_press") -> VolumeScoreResponse:
        """Current volume score and weekly history."""
        workouts = repository.all_workouts(db, exercise)
        recent_rirs = [s.rir for w in workouts[-5:] for s in w.sets][-20:]
        current = calculate_volume_score(recent_rirs, frequency=3, exercise=exercise)

        if current < 30:
            interp = "Low volume — likely undertraining. Consider adding sets or sessions."
        elif current < 60:
            interp = "Moderate volume — productive training load. Keep it up!"
        elif current < 80:
            interp = "High volume — effective but watch for fatigue accumulation."
        else:
            interp = "Very high volume — potential overreaching. Consider reducing intensity."

        weeks: dict[str, list[int]] = {}
        for w in workouts:
            weeks.setdefault(w.created_at.strftime("%Y-W%U"), []).extend(s.rir for s in w.sets)
        weekly = [
            WeeklyScore(week=week, score=calculate_volume_score(rirs, 3, exercise))
            for week, rirs in sorted(weeks.items())
        ]
        return VolumeScoreResponse(current_score=current, interpretation=interp, weekly_scores=weekly[-8:])

    @app.get("/predicted-vs-actual/{exercise}", response_model=list[PredictedVsActual])
    def get_predicted_vs_actual(exercise: Exercise, db: DB) -> list[PredictedVsActual]:
        return [
            PredictedVsActual(
                date=p.created_at.date().isoformat(),
                predicted_reps=p.predicted_reps,
                actual_reps=p.actual_reps,
                weight=p.weight,
            )
            for p in repository.predictions_for(db, exercise)
        ]

    @app.get("/alerts/{exercise}", response_model=list[Alert])
    def get_alerts(exercise: Exercise, db: DB) -> list[Alert]:
        history = repository.session_history(db, exercise)
        context = compute_context(history, exercise)
        if context is None:
            return []
        alerts = predictor.generate_alerts(
            [s.top_weight for s in history], [s.top_rir for s in history], context.volume_score
        )
        return [Alert(**a) for a in alerts]


app = create_app()
