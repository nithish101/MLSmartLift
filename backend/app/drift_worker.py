"""
Scheduled drift detection.

    python -m app.drift_worker

Runs ``run_drift_check`` immediately and then every DRIFT_INTERVAL_SECONDS with
APScheduler, against whichever model version is currently in the serving stage.
Results go to the ``drift_results`` table; PSI gauges are exported for
Prometheus on DRIFT_METRICS_PORT.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from apscheduler.schedulers.base import BaseScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from mlflow import MlflowClient
from prometheus_client import start_http_server
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.db import make_engine, make_session_factory
from app.drift import DriftReport, run_drift_check
from app.observability import configure_logging
from app.registry import latest_in_stage

logger = logging.getLogger(__name__)

JOB_ID = "drift-check"


def check_once(settings: Settings, session_factory: sessionmaker[Session], client: MlflowClient) -> DriftReport:
    ref = latest_in_stage(client, settings.model_name, settings.model_stage)
    return run_drift_check(
        session_factory,
        ref.version if ref else None,
        ref.run_id if ref else None,
        window=timedelta(hours=settings.drift_window_hours),
        min_samples=settings.drift_min_samples,
        max_samples=settings.drift_max_samples,
        threshold=settings.drift_psi_threshold,
    )


def schedule(scheduler: BaseScheduler, settings: Settings, session_factory: sessionmaker[Session]) -> None:
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    def job() -> None:
        try:
            check_once(settings, session_factory, client)
        except Exception:
            logger.exception("drift check failed")

    scheduler.add_job(
        job,
        "interval",
        seconds=settings.drift_interval_seconds,
        id=JOB_ID,
        next_run_time=datetime.now(UTC),  # run once at startup too
        max_instances=1,
        coalesce=True,
        # Run late rather than skip: by default a run delayed >1s (host pause, GC, busy node) is dropped.
        misfire_grace_time=None,
    )


def main() -> None:  # pragma: no cover - process entrypoint
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = make_engine(settings.database_url, pool_size=2, max_overflow=0)
    start_http_server(settings.drift_metrics_port)
    scheduler = BlockingScheduler(timezone="UTC")
    schedule(scheduler, settings, make_session_factory(engine))
    logger.info(
        "drift worker started",
        extra={"interval_seconds": settings.drift_interval_seconds, "window_hours": settings.drift_window_hours},
    )
    scheduler.start()


if __name__ == "__main__":
    main()
