"""Registry-backed serving: stage resolution, promotion and hot reload."""

import time
import uuid

import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from app import registry
from app.config import Settings
from app.features import ModelInput
from app.main import create_app
from app.ml.train import run_training
from app.model_service import ModelService
from tests.conftest import register

pytestmark = pytest.mark.integration

BODY = {"exercise": "squat", "weight": 225, "rir": 2, "bodyweight": 180}


@pytest.fixture
def model_name() -> str:
    return f"smartlift-promo-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def client_mlflow(mlflow_uri):
    return MlflowClient(tracking_uri=mlflow_uri)


def test_latest_in_stage_missing_model(client_mlflow):
    assert registry.latest_in_stage(client_mlflow, "never-registered", "Production") is None


def test_promotion_archives_previous_and_sets_alias(mlflow_uri, client_mlflow, model_name, trained, training_frame):
    v1 = register(mlflow_uri, model_name, trained, training_frame, "Production")
    v2 = register(mlflow_uri, model_name, trained, training_frame, "Staging")
    assert registry.latest_in_stage(client_mlflow, model_name, "Staging").version == v2.version

    registry.promote(client_mlflow, model_name, v2.version, "Production")

    stages = {r.version: r.stage for r in registry.list_versions(client_mlflow, model_name)}
    assert stages == {v1.version: "Archived", v2.version: "Production"}
    assert client_mlflow.get_model_version_by_alias(model_name, "production").version == str(v2.version)
    aliases = client_mlflow.get_registered_model(model_name).aliases
    assert "staging" not in aliases  # v2 left Staging, so its alias went with it


def test_promote_rejects_unknown_stage(client_mlflow):
    with pytest.raises(ValueError, match="stage must be one of"):
        registry.promote(client_mlflow, "x", 1, "Prod")


def test_service_swaps_on_promotion_without_restart(mlflow_uri, client_mlflow, model_name, trained, training_frame):
    v1 = register(mlflow_uri, model_name, trained, training_frame, "Production")
    service = ModelService(mlflow_uri, model_name, "Production")
    assert service.refresh() is True
    assert service.current.ref.version == v1.version
    assert service.refresh() is False  # unchanged stage -> no reload

    v2 = register(mlflow_uri, model_name, trained, training_frame, "Staging")
    assert service.refresh() is False  # Staging doesn't affect a Production server

    registry.promote(client_mlflow, model_name, v2.version, "Production")
    assert service.refresh() is True
    assert service.current.ref.version == v2.version
    assert len(service.predict([ModelInput(**BODY)])) == 1


def test_api_picks_up_promotion_via_polling(
    database_url, redis_url, redis_client, mlflow_uri, client_mlflow, model_name, trained, training_frame
):
    register(mlflow_uri, model_name, trained, training_frame, "Production")
    v2 = register(mlflow_uri, model_name, trained, training_frame, "Staging")
    settings = Settings(
        database_url=database_url,
        redis_url=redis_url,
        mlflow_tracking_uri=mlflow_uri,
        model_name=model_name,
        model_refresh_seconds=0.2,
    )
    with TestClient(create_app(settings)) as c:
        assert c.get("/version").json()["model_version"] == 1
        registry.promote(client_mlflow, model_name, v2.version, "Production")
        deadline = time.monotonic() + 15
        while c.get("/version").json()["model_version"] != v2.version:
            assert time.monotonic() < deadline, "API did not load the promoted version"
            time.sleep(0.2)
        assert c.post("/predict", json=BODY).json()["model_version"] == v2.version


def test_refresh_failure_keeps_serving(mlflow_uri, model_name, trained, training_frame):
    register(mlflow_uri, model_name, trained, training_frame, "Production")
    service = ModelService(mlflow_uri, model_name, "Production")
    service.refresh()
    service._client = MlflowClient(tracking_uri="http://127.0.0.1:9")  # unreachable registry
    assert service.safe_refresh() is False
    assert service.loaded


def test_training_cli_registers_and_bootstrap_is_idempotent(mlflow_uri, client_mlflow, model_name):
    kwargs = {"tracking_uri": mlflow_uri, "experiment": "smartlift-tests", "model_name": model_name, "num_lifters": 4}
    ref = run_training(**kwargs, bootstrap=True, params=None)
    assert ref is not None
    assert ref.stage == "Production"

    run = client_mlflow.get_run(ref.run_id)
    assert {"mae", "rmse", "r2"} <= set(run.data.metrics)
    assert run.data.params["n_estimators"] == "200"

    assert run_training(**kwargs, bootstrap=True) is None  # already in Production
    staged = run_training(**kwargs, stage="Staging")
    assert staged.version == ref.version + 1
    assert staged.stage == "Staging"


def test_registry_cli(mlflow_uri, model_name, trained, training_frame, monkeypatch, capsys):
    v1 = register(mlflow_uri, model_name, trained, training_frame)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_uri)
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        registry.main(["--model-name", model_name, "promote", "--version", str(v1.version), "--stage", "Staging"])
        registry.main(["--model-name", model_name, "list"])
    finally:
        get_settings.cache_clear()
    out = capsys.readouterr().out
    assert f"{model_name} v{v1.version} -> Staging" in out
    assert f"{model_name} v{v1.version}  Staging" in out
