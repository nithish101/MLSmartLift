"""
Shared fixtures.

Integration tests run against a real PostgreSQL started with testcontainers
(requires Docker). Set TEST_DATABASE_URL to reuse an existing database instead.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import Settings
from app.db import make_engine
from app.main import create_app
from app.ml.generate_data import generate_dataset
from app.ml.train import train

FAST_PARAMS = {"n_estimators": 60, "max_depth": 4}


@pytest.fixture(scope="session")
def training_frame():
    return generate_dataset(num_lifters=12, seed=7)


@pytest.fixture(scope="session")
def trained(training_frame):
    return train(training_frame, params=FAST_PARAMS)


@pytest.fixture(scope="session")
def model_path(trained, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(trained.pipeline, path)
    return path


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    if url := os.environ.get("TEST_DATABASE_URL"):
        yield url
        return
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine", driver="psycopg") as pg:
        yield pg.get_connection_url()


@pytest.fixture
def settings(database_url: str, model_path: Path) -> Settings:
    return Settings(database_url=database_url, model_path=str(model_path))


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
    engine = make_engine(settings.database_url)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE predictions, sets, workouts RESTART IDENTITY CASCADE"))
    engine.dispose()
