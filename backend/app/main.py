"""
SmartLift FastAPI backend.
AI-powered strength training assistant API.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, suppress
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import repository
from app.cache import Cache
from app.config import Settings, get_settings
from app.db import Prediction, Workout, WorkoutSet, make_engine, make_session_factory
from app.drift import latest_results
from app.features import Exercise, ModelInput, calculate_volume_score
from app.inference import Inference
from app.inference_log import InferenceLogger
from app.ml import predictor
from app.model_service import ModelService, ModelUnavailableError
from app.observability import RequestContextMiddleware, configure_logging
from app.schemas import (
    Alert,
    CacheStatsEntry,
    DriftFeature,
    DriftResponse,
    HealthResponse,
    PredictedVsActual,
    PredictRequest,
    PredictResponse,
    ProgressPoint,
    Recommendation,
    RecoveryEstimate,
    SetResponse,
    VersionResponse,
    VolumeScoreResponse,
    WeeklyScore,
    WorkoutCreate,
    WorkoutResponse,
)

logger = logging.getLogger("smartlift.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url, settings.db_pool_size, settings.db_max_overflow)
        model_service = ModelService(settings.mlflow_tracking_uri, settings.model_name, settings.model_stage)
        await asyncio.to_thread(model_service.safe_refresh)
        app.state.engine = engine
        app.state.session_factory = make_session_factory(engine)
        app.state.model_service = model_service
        app.state.settings = settings
        cache = Cache(settings.redis_url)
        inference_log = InferenceLogger(
            app.state.session_factory, settings.inference_log_flush_seconds, settings.inference_log_buffer_size
        )
        inference_log.start()
        app.state.inference = Inference(
            model_service,
            cache,
            settings.prediction_cache_ttl_seconds,
            settings.feature_cache_ttl_seconds,
            inference_log,
        )
        poller = asyncio.create_task(model_service.poll(settings.model_refresh_seconds))
        yield
        poller.cancel()
        with suppress(asyncio.CancelledError):
            await poller
        await asyncio.to_thread(inference_log.stop)
        cache.close()
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
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestContextMiddleware)
    register_routes(app)
    return app


# ─── Dependencies ────────────────────────────────────────────────────────────


def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


def get_model(request: Request) -> ModelService:
    service: ModelService = request.app.state.model_service
    return service


def get_inference(request: Request) -> Inference:
    inference: Inference = request.app.state.inference
    return inference


DB = Annotated[Session, Depends(get_db)]
Model = Annotated[ModelService, Depends(get_model)]
Infer = Annotated[Inference, Depends(get_inference)]


def _serialize_workout(w: Workout) -> WorkoutResponse:
    rirs = [s.rir for s in w.sets]
    return WorkoutResponse(
        id=w.id,
        exercise=w.exercise,
        bodyweight=w.bodyweight,
        sets=[SetResponse(set_number=s.set_number, weight=s.weight, reps=s.reps, rir=s.rir) for s in w.sets],
        recovery=RecoveryEstimate(**predictor.estimate_recovery_time(rirs, w.exercise)),
        created_at=w.created_at.isoformat(),
    )


def _model_unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail="Model not loaded")


# ─── Routes ──────────────────────────────────────────────────────────────────


def register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    def health(db: DB, model: Model, inference: Infer) -> HealthResponse:
        try:
            db.execute(text("SELECT 1"))
            database = "ok"
        except Exception:
            database = "unavailable"
        status = "ok" if database == "ok" and model.loaded else "degraded"
        current = model.current
        return HealthResponse(
            status=status,
            database=database,
            model_loaded=current is not None,
            model_version=current.ref.version if current else None,
            cache="ok" if inference.cache.ping() else "unavailable",
            cache_stats={
                name: CacheStatsEntry.model_validate(entry) for name, entry in inference.cache.stats.snapshot().items()
            },
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/version", response_model=VersionResponse)
    def version(request: Request, model: Model) -> VersionResponse:
        """The app version and the registry model version currently serving."""
        settings: Settings = request.app.state.settings
        current = model.current
        return VersionResponse(
            app_version=settings.app_version,
            model_name=model.model_name,
            model_stage=model.stage,
            model_version=current.ref.version if current else None,
            run_id=current.ref.run_id if current else None,
            loaded_at=current.loaded_at.isoformat() if current else None,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, inference: Infer) -> PredictResponse:
        """Predict reps for one feature vector (served from cache when seen before)."""
        try:
            result = inference.predict(req)
        except ModelUnavailableError as exc:
            raise _model_unavailable() from exc
        return PredictResponse(
            predicted_reps=result.reps,
            raw_prediction=result.raw,
            outcome=result.outcome,
            model_version=result.model_version,
            cached=result.cached,
        )

    @app.post("/workouts", response_model=WorkoutResponse)
    def create_workout(workout: WorkoutCreate, db: DB, model: Model, inference: Infer) -> WorkoutResponse:
        """Log a new workout with sets."""
        # History features as of before this session: what the model would have known going in.
        context = inference.history_features(db, workout.exercise).context
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
            history = (
                {
                    "num_sets": context.num_sets,
                    "frequency": context.frequency,
                    "volume_score": context.volume_score,
                    "progression_trend": context.progression_trend,
                }
                if context
                else {}
            )
            features = ModelInput(
                exercise=workout.exercise, weight=top.weight, rir=top.rir, bodyweight=workout.bodyweight, **history
            )
            predicted = inference.predict(features, source="workout").reps
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
        inference.invalidate_features(workout.exercise)
        logger.info(
            "workout logged",
            extra={"workout_id": row.id, "exercise": workout.exercise, "sets": len(workout.sets)},
        )
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
        inference: Infer,
        bodyweight: Annotated[float, Query(ge=50, le=700)] = 180,
    ) -> Recommendation:
        """AI recommendation for the next session."""
        context = inference.history_features(db, exercise).context
        if context is None:
            return Recommendation(
                exercise=exercise,
                recommended_weight=predictor.DEFAULT_START_WEIGHT[exercise],
                expected_reps=8,
                target_rir=predictor.TARGET_RIR,
                confidence=0.5,
            )
        try:
            rec = predictor.recommend_next_session(inference.pinned_model(), exercise, bodyweight, context)
        except ModelUnavailableError as exc:
            raise _model_unavailable() from exc
        return Recommendation(**rec)

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

    @app.get("/drift", response_model=DriftResponse)
    def get_drift(db: DB) -> DriftResponse:
        """Per-feature PSI from the most recent drift check."""
        rows = latest_results(db)
        if not rows:
            return DriftResponse(computed_at=None, model_version=None, sample_size=0, features=[])
        first = rows[0]
        return DriftResponse(
            computed_at=first.computed_at.isoformat(),
            model_version=first.model_version,
            sample_size=first.sample_size,
            features=[
                DriftFeature(feature=r.feature, psi=r.psi, threshold=r.threshold, drifted=r.drifted) for r in rows
            ],
        )

    @app.get("/alerts/{exercise}", response_model=list[Alert])
    def get_alerts(exercise: Exercise, db: DB, inference: Infer) -> list[Alert]:
        features = inference.history_features(db, exercise)
        if features.context is None:
            return []
        alerts = predictor.generate_alerts(features.top_weights, features.top_rirs, features.context.volume_score)
        return [Alert(**a) for a in alerts]


app = create_app()
