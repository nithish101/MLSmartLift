"""Inference logging, the PSI drift job against real Postgres/MLflow, and the scheduler."""

import io
import json
import logging
import time
from datetime import UTC, datetime, timedelta

import mlflow
import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import func, insert, select, text

from app.db import DriftResult, InferenceLog, make_engine, make_session_factory
from app.drift import load_reference, run_drift_check
from app.drift_worker import JOB_ID, check_once, schedule
from app.features import FEATURE_COLUMNS
from app.metrics import FEATURE_PSI
from app.observability import JsonFormatter

pytestmark = pytest.mark.integration

NOW = datetime(2026, 9, 1, 12, tzinfo=UTC)


@pytest.fixture
def session_factory(database_url):
    engine = make_engine(database_url)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE inference_log, drift_results RESTART IDENTITY"))
    yield make_session_factory(engine)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE inference_log, drift_results RESTART IDENTITY"))
    engine.dispose()


def insert_live(session_factory, frame, model_version, when=NOW):
    rows = [
        {
            **{c: r[c] for c in FEATURE_COLUMNS},
            "created_at": when - timedelta(minutes=5),
            "model_version": model_version,
            "source": "predict",
            "predicted": 7.0,
            "cached": False,
        }
        for r in frame.to_dict("records")
    ]
    with session_factory() as db:
        db.execute(insert(InferenceLog), rows)
        db.commit()


def check(session_factory, production_model, **overrides):
    kwargs = {"window": timedelta(hours=1), "min_samples": 100, "max_samples": 10_000, "threshold": 0.2, "now": NOW}
    kwargs.update(overrides)
    return run_drift_check(session_factory, production_model.version, production_model.run_id, **kwargs)


def test_reference_is_logged_with_the_model(mlflow_uri, production_model):
    mlflow.set_tracking_uri(mlflow_uri)
    reference = load_reference(production_model.run_id)
    assert reference is not None
    assert set(reference["features"]) == set(FEATURE_COLUMNS)


def test_served_predictions_are_logged(client, settings):
    body = {"exercise": "squat", "weight": 225, "rir": 2, "bodyweight": 180}
    client.post("/predict", json=body)
    client.post("/predict", json=body)  # cache hit is still a served prediction
    client.post(
        "/workouts", json={"exercise": "squat", "bodyweight": 180, "sets": [{"weight": 225, "reps": 5, "rir": 2}]}
    )
    client.get("/recommendation/squat")
    client.app.state.inference.inference_log.flush()

    engine = make_engine(settings.database_url)
    with make_session_factory(engine)() as db:
        rows = db.scalars(select(InferenceLog).order_by(InferenceLog.id)).all()
    engine.dispose()
    assert [r.cached for r in rows[:2]] == [False, True]
    assert {r.source for r in rows} == {"predict", "workout", "recommendation"}
    assert all(r.model_version == 1 and r.exercise == "squat" for r in rows)
    assert rows[0].weight == 225
    assert rows[0].num_sets == 4


def test_stable_traffic_reports_ok(session_factory, production_model, training_frame):
    insert_live(session_factory, training_frame.sample(1500, random_state=3, replace=True), production_model.version)
    report = check(session_factory, production_model)
    assert report.status == "ok"
    assert report.sample_size == 1500
    assert report.drifted == []
    with session_factory() as db:
        assert db.scalar(select(func.count()).select_from(DriftResult)) == len(FEATURE_COLUMNS)


def test_shifted_traffic_is_flagged_persisted_logged_and_exported(session_factory, production_model, training_frame):
    live = training_frame.sample(1500, random_state=4, replace=True).copy()
    live["bodyweight"] = live["bodyweight"] + 60  # much heavier lifters than in training
    insert_live(session_factory, live, production_model.version)

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logging.getLogger("app.drift").addHandler(handler)
    try:
        report = check(session_factory, production_model)
    finally:
        logging.getLogger("app.drift").removeHandler(handler)

    assert report.status == "drift"
    assert report.drifted == ["bodyweight"]
    assert report.psi["bodyweight"] > 0.2

    with session_factory() as db:
        row = db.scalar(select(DriftResult).where(DriftResult.feature == "bodyweight"))
    assert row.drifted
    assert row.psi == report.psi["bodyweight"]
    assert row.sample_size == 1500
    assert row.model_version == production_model.version
    assert row.window_end - row.window_start == timedelta(hours=1)

    warnings = [json.loads(line) for line in buf.getvalue().splitlines()]
    flagged = [w for w in warnings if w["level"] == "WARNING"]
    assert [w["feature"] for w in flagged] == ["bodyweight"]
    assert flagged[0]["psi"] == report.psi["bodyweight"]

    assert FEATURE_PSI.labels("bodyweight")._value.get() == report.psi["bodyweight"]


def test_window_excludes_old_rows_and_other_versions(session_factory, production_model, training_frame):
    insert_live(
        session_factory,
        training_frame.sample(500, random_state=5, replace=True),
        production_model.version,
        NOW - timedelta(days=2),
    )
    insert_live(
        session_factory, training_frame.sample(500, random_state=6, replace=True), production_model.version + 99
    )
    report = check(session_factory, production_model)
    assert report.status == "insufficient_data"
    assert report.sample_size == 0


def test_insufficient_samples_writes_nothing(session_factory, production_model, training_frame):
    insert_live(session_factory, training_frame.sample(20, random_state=7, replace=True), production_model.version)
    assert check(session_factory, production_model).status == "insufficient_data"
    with session_factory() as db:
        assert db.scalar(select(func.count()).select_from(DriftResult)) == 0


def test_no_model_and_no_reference(session_factory, mlflow_uri):
    assert (
        run_drift_check(
            session_factory, None, None, window=timedelta(hours=1), min_samples=1, max_samples=10, threshold=0.2
        ).status
        == "no_model"
    )
    mlflow.set_tracking_uri(mlflow_uri)
    with mlflow.start_run() as run:  # a run that never logged a reference
        pass
    report = run_drift_check(
        session_factory, 1, run.info.run_id, window=timedelta(hours=1), min_samples=1, max_samples=10, threshold=0.2
    )
    assert report.status == "no_reference"


def test_drift_endpoint_returns_latest_check(client, settings, production_model, training_frame):
    assert client.get("/drift").json() == {"computed_at": None, "model_version": None, "sample_size": 0, "features": []}
    engine = make_engine(settings.database_url)
    factory = make_session_factory(engine)
    insert_live(factory, training_frame.sample(300, random_state=8, replace=True), production_model.version)
    check(factory, production_model, now=NOW - timedelta(minutes=1))  # an older check
    check(factory, production_model)
    engine.dispose()
    body = client.get("/drift").json()
    assert body["computed_at"].startswith("2026-09-01T12:00:00")
    assert body["model_version"] == production_model.version
    assert {f["feature"] for f in body["features"]} == set(FEATURE_COLUMNS)


def test_scheduler_runs_check_against_production(settings, session_factory, production_model, training_frame):
    insert_live(
        session_factory,
        training_frame.sample(300, random_state=9, replace=True),
        production_model.version,
        datetime.now(UTC),
    )
    settings.drift_min_samples = 100
    settings.drift_interval_seconds = 3600

    from mlflow import MlflowClient

    report = check_once(settings, session_factory, MlflowClient(tracking_uri=settings.mlflow_tracking_uri))
    assert report.status == "ok"
    assert report.model_version == production_model.version

    scheduler = BackgroundScheduler(timezone="UTC")
    schedule(scheduler, settings, session_factory)
    assert scheduler.get_job(JOB_ID).trigger.interval == timedelta(seconds=3600)
    assert scheduler.get_job(JOB_ID).misfire_grace_time is None  # late runs still happen
    scheduler.start()
    try:
        deadline = time.monotonic() + 30
        while True:
            with session_factory() as db:
                if db.scalar(select(func.count(func.distinct(DriftResult.computed_at)))) >= 2:
                    break
            assert time.monotonic() < deadline, "scheduled drift check did not run"
            time.sleep(0.2)
    finally:
        scheduler.shutdown(wait=True)


def test_workout_prediction_uses_history_features(client, settings):
    """The prediction stored for a logged workout uses the lifter's history, not defaults."""
    for weight in (135, 140, 145):
        sets = [{"weight": weight, "reps": 8, "rir": 2}] * 5
        client.post("/workouts", json={"exercise": "bench_press", "bodyweight": 180, "sets": sets})
    client.app.state.inference.inference_log.flush()

    engine = make_engine(settings.database_url)
    with make_session_factory(engine)() as db:
        rows = db.scalars(select(InferenceLog).where(InferenceLog.source == "workout").order_by(InferenceLog.id)).all()
    engine.dispose()
    assert len(rows) == 3
    first, _, last = rows
    assert (first.num_sets, first.frequency) == (4, 3)  # no history yet: defaults
    assert last.num_sets == 5  # average sets of the previous sessions
    assert last.frequency == 2  # the two earlier sessions this week
    assert last.progression_trend == pytest.approx(1.0)  # +5 lb per session
