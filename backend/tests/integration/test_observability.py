"""/metrics content and request-id propagation through a real request."""

import io
import json
import logging
import re

import pytest

from app.observability import JsonFormatter, RequestIdFilter

pytestmark = pytest.mark.integration

BODY = {"exercise": "squat", "weight": 245, "rir": 1, "bodyweight": 190}


def metric(text: str, name: str, **labels: str) -> float:
    """Value of one sample from Prometheus exposition text (0 if absent)."""
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        m = re.match(r"^([a-zA-Z_:]+)(\{(.*)\})? (.+)$", line)
        if not m or m.group(1) != name:
            continue
        found = dict(re.findall(r'(\w+)="([^"]*)"', m.group(3) or ""))
        if all(found.get(k) == v for k, v in labels.items()):
            return float(m.group(4))
    return 0.0


def test_metrics_endpoint_exposes_all_series(client, production_model):
    client.post("/predict", json=BODY)
    client.post("/predict", json=BODY)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    text = resp.text
    for name in (
        "smartlift_http_request_duration_seconds_bucket",
        "smartlift_inference_duration_seconds_bucket",
        "smartlift_cache_requests_total",
        "smartlift_cache_hit_ratio",
        "smartlift_model_version",
        "smartlift_predictions_total",
    ):
        assert name in text, name
    assert metric(text, "smartlift_model_version", model_name=production_model.name, stage="Production") == 1


def test_metrics_move_with_traffic(client):
    before = client.get("/metrics").text
    client.post("/predict", json=BODY)  # miss
    client.post("/predict", json=BODY)  # hit
    after = client.get("/metrics").text

    def delta(name, **labels):
        return metric(after, name, **labels) - metric(before, name, **labels)

    assert delta("smartlift_http_request_duration_seconds_count", method="POST", route="/predict", status="200") == 2
    assert delta("smartlift_inference_duration_seconds_count", cache="miss") == 1
    assert delta("smartlift_inference_duration_seconds_count", cache="hit") == 1
    assert delta("smartlift_cache_requests_total", cache="prediction", result="hit") == 1
    assert delta("smartlift_cache_requests_total", cache="prediction", result="miss") == 1
    outcome = client.post("/predict", json=BODY).json()["outcome"]
    assert metric(client.get("/metrics").text, "smartlift_predictions_total", outcome=outcome) >= 3
    assert 0 < metric(after, "smartlift_cache_hit_ratio", cache="prediction") <= 1


def test_unmatched_routes_share_one_label(client):
    client.get("/no/such/path/123")
    client.get("/no/such/path/456")
    text = client.get("/metrics").text
    assert 'route="/no/such/path/123"' not in text
    assert metric(text, "smartlift_http_request_duration_seconds_count", route="unmatched", status="404") >= 2


def test_path_params_use_template_label(client):
    client.get("/progress/squat")
    text = client.get("/metrics").text
    assert metric(text, "smartlift_http_request_duration_seconds_count", route="/progress/{exercise}") >= 1


@pytest.fixture
def log_lines():
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    yield lambda: [json.loads(line) for line in buf.getvalue().splitlines()]
    root.removeHandler(handler)


def test_request_id_propagates_through_request(client, log_lines):
    sets = [{"weight": 225, "reps": 5, "rir": 2}]
    resp = client.post(
        "/workouts", json={"exercise": "squat", "bodyweight": 180, "sets": sets}, headers={"X-Request-ID": "trace-123"}
    )
    assert resp.headers["x-request-id"] == "trace-123"

    lines = [entry for entry in log_lines() if entry["request_id"] == "trace-123"]
    by_logger = {entry["logger"]: entry for entry in lines}
    # Logged inside the (threadpool) endpoint and by the access-log middleware.
    assert by_logger["smartlift.api"]["message"] == "workout logged"
    assert by_logger["smartlift.api"]["workout_id"] == resp.json()["id"]
    access = by_logger["smartlift.access"]
    assert access["route"] == "/workouts"
    assert access["status"] == 200
    assert access["duration_ms"] > 0


def test_request_id_generated_when_absent_or_invalid(client, log_lines):
    generated = client.get("/health").headers["x-request-id"]
    assert re.fullmatch(r"[0-9a-f]{32}", generated)
    replaced = client.get("/health", headers={"X-Request-ID": "has spaces"}).headers["x-request-id"]
    assert replaced != "has spaces"
    assert {e["request_id"] for e in log_lines() if e["logger"] == "smartlift.access"} >= {generated, replaced}
