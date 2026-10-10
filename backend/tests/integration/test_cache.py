"""Redis caching against a real Redis: cold path, warm path, TTLs, invalidation, outage."""

import pytest
from fastapi.testclient import TestClient

from app.cache import Cache, features_key
from app.config import Settings
from app.main import create_app

pytestmark = pytest.mark.integration

BODY = {"exercise": "deadlift", "weight": 315, "rir": 2, "bodyweight": 180}


def stats(client, cache):
    return client.get("/health").json()["cache_stats"].get(cache, {"hits": 0, "misses": 0})


def test_cold_cache_computes_then_warm_cache_serves(client, redis_client, settings):
    assert redis_client.dbsize() == 0  # cold

    first = client.post("/predict", json=BODY).json()
    assert first["cached"] is False
    assert stats(client, "prediction") == {"hits": 0, "misses": 1, "hit_ratio": 0.0}

    [key] = redis_client.keys("smartlift:pred:*")
    assert key.startswith(f"smartlift:pred:v{first['model_version']}:")
    assert 0 < redis_client.ttl(key) <= settings.prediction_cache_ttl_seconds

    second = client.post("/predict", json=BODY).json()
    assert second["cached"] is True
    assert second["raw_prediction"] == first["raw_prediction"]
    assert stats(client, "prediction")["hits"] == 1


def test_distinct_vectors_are_distinct_entries(client, redis_client):
    client.post("/predict", json=BODY)
    client.post("/predict", json={**BODY, "weight": 325})
    assert len(redis_client.keys("smartlift:pred:*")) == 2


def test_expired_entry_is_recomputed(client, redis_client):
    client.post("/predict", json=BODY)
    for key in redis_client.keys("smartlift:pred:*"):
        redis_client.delete(key)
    assert client.post("/predict", json=BODY).json()["cached"] is False


def test_feature_lookup_cached_and_invalidated_on_new_workout(client, redis_client, settings):
    sets = [{"weight": 135, "reps": 8, "rir": 2}] * 3
    client.post("/workouts", json={"exercise": "bench_press", "bodyweight": 180, "sets": sets})

    before = stats(client, "features")  # logging the workout itself looked up (then invalidated) history
    cold = client.get("/recommendation/bench_press").json()
    after = stats(client, "features")
    assert (after["hits"] - before["hits"], after["misses"] - before["misses"]) == (0, 1)  # cold after invalidation
    key = features_key("bench_press")
    assert 0 < redis_client.ttl(key) <= settings.feature_cache_ttl_seconds

    warm = client.get("/recommendation/bench_press").json()
    assert warm == cold
    client.get("/alerts/bench_press")
    assert stats(client, "features")["hits"] - before["hits"] == 2

    heavier = [{"weight": 185, "reps": 8, "rir": 2}] * 3
    client.post("/workouts", json={"exercise": "bench_press", "bodyweight": 180, "sets": heavier})
    assert redis_client.exists(key) == 0  # invalidated
    after = client.get("/recommendation/bench_press").json()
    assert after["recommended_weight"] >= 180  # reflects the new session, not the cached one


def test_recommendation_predictions_hit_prediction_cache(client):
    sets = [{"weight": 225, "reps": 6, "rir": 2}] * 3
    client.post("/workouts", json={"exercise": "squat", "bodyweight": 180, "sets": sets})
    client.get("/recommendation/squat")
    misses = stats(client, "prediction")["misses"]
    client.get("/recommendation/squat")
    after = stats(client, "prediction")
    assert after["misses"] == misses
    assert after["hits"] >= 1


def test_redis_outage_degrades_to_uncached(database_url, mlflow_uri, production_model, redis_client):
    settings = Settings(
        database_url=database_url,
        mlflow_tracking_uri=mlflow_uri,
        model_name=production_model.name,
        redis_url="redis://127.0.0.1:1/0",  # nothing listening
    )
    with TestClient(create_app(settings)) as c:
        health = c.get("/health").json()
        assert health["cache"] == "unavailable"
        assert health["status"] == "ok"  # cache is optional
        for _ in range(2):
            resp = c.post("/predict", json=BODY)
            assert resp.status_code == 200
            assert resp.json()["cached"] is False
        assert c.get("/health").json()["cache_stats"]["prediction"]["misses"] == 2


def test_cache_helpers(redis_url, redis_client):
    cache = Cache(redis_url)
    assert cache.ping()
    assert cache.get_json("x", "k") is None
    cache.set_json("k", {"a": 1}, ttl_seconds=60)
    assert cache.get_json("x", "k") == {"a": 1}
    assert 0 < cache.ttl("k") <= 60
    cache.delete("k")
    assert cache.get_json("x", "k") is None
    cache.close()
