"""End-to-end API tests against real PostgreSQL and the trained model."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.main import create_app
from app.seed import DEMO_WORKOUTS, seed

pytestmark = pytest.mark.integration

PREDICT_BODY = {
    "exercise": "bench_press",
    "weight": 155,
    "rir": 2,
    "bodyweight": 180,
    "num_sets": 4,
    "frequency": 3,
    "volume_score": 45,
    "progression_trend": 0.6,
}


def log(client: TestClient, exercise: str, weight: float, reps: int = 8, rir: int = 2, n_sets: int = 3):
    body = {
        "exercise": exercise,
        "bodyweight": 180,
        "sets": [{"weight": weight, "reps": reps, "rir": rir} for _ in range(n_sets)],
    }
    resp = client.post("/workouts", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_health_reports_db_and_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "ok", "model_loaded": True}


class TestPredict:
    def test_end_to_end_prediction(self, client, trained):
        resp = client.post("/predict", json=PREDICT_BODY)
        assert resp.status_code == 200
        body = resp.json()
        assert 1 <= body["predicted_reps"] <= 15
        assert body["outcome"] in {"decrease", "maintain", "increase"}
        # Served prediction matches the trained pipeline exactly.
        from app.features import ModelInput, to_frame

        expected = float(trained.pipeline.predict(to_frame([ModelInput(**PREDICT_BODY)]))[0])
        assert body["raw_prediction"] == pytest.approx(expected)

    def test_defaults_for_history_features(self, client):
        body = {k: PREDICT_BODY[k] for k in ("exercise", "weight", "rir", "bodyweight")}
        assert client.post("/predict", json=body).status_code == 200

    @pytest.mark.parametrize(
        "patch",
        [{"exercise": "curl"}, {"weight": -5}, {"rir": 9}, {"bodyweight": 10}, {"volume_score": 500}, {"extra": 1}],
    )
    def test_invalid_input_is_422(self, client, patch):
        resp = client.post("/predict", json={**PREDICT_BODY, **patch})
        assert resp.status_code == 422

    def test_missing_field_is_422(self, client):
        resp = client.post("/predict", json={"exercise": "squat", "weight": 200})
        assert resp.status_code == 422


class TestWorkouts:
    def test_log_and_list(self, client):
        created = log(client, "squat", 225, reps=6, rir=-1)
        assert created["id"] == 1
        assert [s["set_number"] for s in created["sets"]] == [1, 2, 3]
        assert all(s["rir"] == 0 for s in created["sets"])  # failure stored as 0 RIR
        assert created["recovery"]["recovery_hours"] > 48

        log(client, "bench_press", 135)
        all_rows = client.get("/workouts").json()
        assert [w["exercise"] for w in all_rows] == ["bench_press", "squat"]
        only_squat = client.get("/workouts", params={"exercise": "squat"}).json()
        assert [w["id"] for w in only_squat] == [1]
        assert len(client.get("/workouts", params={"limit": 1}).json()) == 1

    def test_logging_stores_prediction(self, client):
        log(client, "deadlift", 315, reps=5)
        [row] = client.get("/predicted-vs-actual/deadlift").json()
        assert row["actual_reps"] == 5
        assert row["weight"] == 315
        assert row["predicted_reps"] >= 1

    def test_rejects_bad_workout(self, client):
        assert client.post("/workouts", json={"exercise": "squat", "bodyweight": 180, "sets": []}).status_code == 422
        assert (
            client.post(
                "/workouts", json={"exercise": "curl", "bodyweight": 180, "sets": [{"weight": 20, "reps": 5, "rir": 1}]}
            ).status_code
            == 422
        )


class TestRecommendation:
    def test_default_without_history(self, client):
        rec = client.get("/recommendation/deadlift").json()
        assert rec == {
            "exercise": "deadlift",
            "recommended_weight": 135,
            "expected_reps": 8,
            "target_rir": 2,
            "confidence": 0.5,
        }

    def test_uses_logged_history(self, client):
        for w in (135, 140, 145):
            log(client, "bench_press", w)
        rec = client.get("/recommendation/bench_press", params={"bodyweight": 185}).json()
        assert rec["exercise"] == "bench_press"
        assert rec["recommended_weight"] in {140, 145, 150}
        assert rec["recommended_weight"] % 5 == 0
        assert 0.3 <= rec["confidence"] <= 1.0

    def test_unknown_exercise_is_422(self, client):
        assert client.get("/recommendation/curl").status_code == 422


class TestAnalytics:
    @pytest.fixture
    def seeded(self, client, settings):
        engine = make_engine(settings.database_url)
        with make_session_factory(engine)() as db:
            seed(db, client.app.state.model_service, now=datetime(2026, 7, 1, 12, tzinfo=UTC))
        engine.dispose()
        return client

    def test_progress_is_daily_best_e1rm(self, seeded):
        points = seeded.get("/progress/bench_press").json()
        assert len(points) == len(DEMO_WORKOUTS["bench_press"])
        assert [p["date"] for p in points] == sorted(p["date"] for p in points)
        assert points[-1]["estimated_1rm"] > points[0]["estimated_1rm"]

    def test_volume_score(self, seeded):
        body = seeded.get("/volume-score", params={"exercise": "squat"}).json()
        assert 0 <= body["current_score"] <= 100
        assert body["interpretation"]
        assert 1 <= len(body["weekly_scores"]) <= 8

    def test_predicted_vs_actual(self, seeded):
        rows = seeded.get("/predicted-vs-actual/squat").json()
        assert len(rows) == len(DEMO_WORKOUTS["squat"])

    def test_alerts(self, seeded):
        alerts = seeded.get("/alerts/deadlift").json()
        assert all(a["type"] in {"warning", "info", "tip"} for a in alerts)

    def test_alerts_empty_without_history(self, client):
        assert client.get("/alerts/squat").json() == []

    def test_seed_is_idempotent(self, seeded, settings):
        engine = make_engine(settings.database_url)
        with make_session_factory(engine)() as db:
            seed(db, None, now=datetime.now(UTC) - timedelta(days=1))
        engine.dispose()
        total = sum(len(v) for v in DEMO_WORKOUTS.values())
        assert len(seeded.get("/workouts", params={"limit": 500}).json()) == total


def test_service_degrades_without_model(database_url, tmp_path):
    app = create_app(Settings(database_url=database_url, model_path=str(tmp_path / "none.joblib")))
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json=PREDICT_BODY).status_code == 503
        log(c, "squat", 200)  # logging still works without a model
        assert c.get("/recommendation/squat").status_code == 503
        assert c.get("/predicted-vs-actual/squat").json() == []
    engine = make_engine(database_url)
    with engine.begin() as conn:
        from sqlalchemy import text

        conn.execute(text("TRUNCATE predictions, sets, workouts RESTART IDENTITY CASCADE"))
    engine.dispose()
