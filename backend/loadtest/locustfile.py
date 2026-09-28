"""
Load profile for the SmartLift inference API.

    docker compose --profile loadtest run --rm loadtest   # in-network, against http://api:8000

Each user issues ~1 request/second (constant_throughput), so TARGET_RPS users
produce TARGET_RPS requests/second. Traffic ramps up over WARMUP_SECONDS; when
the warmup ends the statistics are reset, so only the steady-state
TEST_SECONDS that follow are measured and gated.

Request bodies are feature vectors drawn from the same synthetic generator the
model is trained on, so traffic looks like real lifters (and doesn't trip the
drift detector). Vectors repeat across users, which exercises the cache at a
realistic hit rate.

Environment: TARGET_RPS (50), WARMUP_SECONDS (30), TEST_SECONDS (120).
"""

from __future__ import annotations

import os
import random
from typing import Any

from locust import HttpUser, LoadTestShape, constant_throughput, events, task

from app.features import FEATURE_COLUMNS
from app.ml.generate_data import generate_dataset

TARGET_RPS = int(os.environ.get("TARGET_RPS", "50"))
WARMUP_SECONDS = int(os.environ.get("WARMUP_SECONDS", "30"))
TEST_SECONDS = int(os.environ.get("TEST_SECONDS", "120"))

# A fixed pool of realistic feature vectors (seed differs from training data).
_POOL: list[dict[str, Any]] = [
    {str(k): (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    for row in generate_dataset(num_lifters=20, seed=2026)[FEATURE_COLUMNS].to_dict("records")
]
EXERCISES = ("bench_press", "squat", "deadlift")


class Lifter(HttpUser):
    wait_time = constant_throughput(1)

    @task(9)
    def predict(self) -> None:
        body = random.choice(_POOL)
        with self.client.post("/predict", json=body, name="/predict", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")
            elif "predicted_reps" not in resp.json():
                resp.failure("missing predicted_reps")

    @task(1)
    def recommendation(self) -> None:
        exercise = random.choice(EXERCISES)
        self.client.get(f"/recommendation/{exercise}", name="/recommendation/{exercise}")


class WarmupThenSteady(LoadTestShape):
    """Linear ramp to TARGET_RPS users during warmup, reset stats, then hold for TEST_SECONDS."""

    def __init__(self) -> None:
        super().__init__()
        self._reset_done = False

    def tick(self) -> tuple[int, float] | None:
        t = self.get_run_time()
        if t < WARMUP_SECONDS:
            users = max(1, round(TARGET_RPS * min(1.0, t / max(WARMUP_SECONDS * 0.5, 1))))
            return users, max(1.0, TARGET_RPS / max(WARMUP_SECONDS * 0.5, 1))
        if not self._reset_done and self.runner is not None:
            self.runner.stats.reset_all()
            self._reset_done = True
            print(f"warmup complete after {WARMUP_SECONDS}s; statistics reset, measuring for {TEST_SECONDS}s")
        if t < WARMUP_SECONDS + TEST_SECONDS:
            return TARGET_RPS, float(TARGET_RPS)
        return None


@events.init.add_listener
def _describe(environment: Any, **_: Any) -> None:
    print(f"load profile: target {TARGET_RPS} rps, warmup {WARMUP_SECONDS}s, measure {TEST_SECONDS}s")
