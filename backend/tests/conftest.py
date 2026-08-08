"""
Shared fixtures.

Integration tests run against real PostgreSQL, Redis and MLflow tracking
server containers started with testcontainers (requires Docker). Set
TEST_DATABASE_URL / TEST_REDIS_URL / TEST_MLFLOW_URI to reuse running services.
"""

from __future__ import annotations

import os
import time
import urllib.request
import uuid
from collections.abc import Iterator

import pytest
import redis
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import Settings
from app.db import make_engine
from app.main import create_app
from app.ml.generate_data import generate_dataset
from app.ml.train import log_and_register, train
from app.registry import ModelRef, promote

MLFLOW_IMAGE = "ghcr.io/mlflow/mlflow:v3.14.0"

FAST_PARAMS = {"n_estimators": 60, "max_depth": 4}


@pytest.fixture(scope="session")
def training_frame():
    return generate_dataset(num_lifters=12, seed=7)


@pytest.fixture(scope="session")
def trained(training_frame):
    return train(training_frame, params=FAST_PARAMS)



@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    if url := os.environ.get("TEST_DATABASE_URL"):
        yield url
        return
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine", driver="psycopg") as pg:
        yield pg.get_connection_url()


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    if url := os.environ.get("TEST_REDIS_URL"):
        yield url
        return
    from testcontainers.community.redis import RedisContainer

    with RedisContainer("redis:7-alpine") as container:
        yield f"redis://{container.get_container_host_ip()}:{container.get_exposed_port(6379)}/0"


@pytest.fixture
def redis_client(redis_url: str) -> Iterator[redis.Redis]:
    client = redis.Redis.from_url(redis_url, decode_responses=True)
    client.flushdb()
    yield client
    client.close()


def _wait_http(url: str, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


@pytest.fixture(scope="session")
def mlflow_uri() -> Iterator[str]:
    if uri := os.environ.get("TEST_MLFLOW_URI"):
        yield uri
        return
    from testcontainers.core.container import DockerContainer

    container = (
        DockerContainer(MLFLOW_IMAGE)
        .with_exposed_ports(5000)
        .with_command(
            "mlflow server --host 0.0.0.0 --port 5000 "
            "--backend-store-uri sqlite:////tmp/mlflow.db "
            "--artifacts-destination /tmp/artifacts --serve-artifacts"
        )
    )
    with container:
        uri = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(5000)}"
        _wait_http(f"{uri}/health")
        yield uri


def register(mlflow_uri: str, model_name: str, trained, frame, stage: str | None = None) -> ModelRef:
    from mlflow import MlflowClient

    ref = log_and_register(trained, frame, tracking_uri=mlflow_uri, experiment="smartlift-tests", model_name=model_name)
    if stage:
        ref = promote(MlflowClient(tracking_uri=mlflow_uri), model_name, ref.version, stage)
    return ref


@pytest.fixture(scope="session")
def production_model(mlflow_uri: str, trained, training_frame) -> ModelRef:
    """The trained test model registered as v1 and promoted to Production."""
    return register(mlflow_uri, f"smartlift-test-{uuid.uuid4().hex[:8]}", trained, training_frame, "Production")


@pytest.fixture
def settings(
    database_url: str, mlflow_uri: str, redis_url: str, redis_client: redis.Redis, production_model: ModelRef
) -> Settings:
    return Settings(
        database_url=database_url,
        redis_url=redis_url,
        mlflow_tracking_uri=mlflow_uri,
        model_name=production_model.name,
        model_refresh_seconds=3600,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
    engine = make_engine(settings.database_url)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE predictions, sets, workouts RESTART IDENTITY CASCADE"))
    engine.dispose()
