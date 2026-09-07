"""Runtime configuration, read from environment variables (see .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "SmartLift API"
    app_version: str = "1.1.0"

    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://smartlift:smartlift@localhost:5432/smartlift"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # MLflow tracking server / model registry.
    mlflow_tracking_uri: str = "http://localhost:5050"
    mlflow_experiment: str = "smartlift"
    model_name: str = "smartlift-reps"
    # Registry stage the API serves; re-checked every model_refresh_seconds.
    model_stage: str = "Production"
    model_refresh_seconds: float = 30.0

    # Redis cache. Predictions are keyed by model version + feature hash;
    # history features are invalidated whenever a workout is logged.
    redis_url: str = "redis://localhost:6379/0"
    prediction_cache_ttl_seconds: int = 3600
    feature_cache_ttl_seconds: int = 300

    # Drift detection. The worker compares served features from the last
    # drift_window_hours with the model's training reference every
    # drift_interval_seconds and flags features whose PSI exceeds the threshold.
    drift_window_hours: float = 24.0
    drift_interval_seconds: int = 300
    drift_min_samples: int = 200
    drift_max_samples: int = 50_000
    drift_psi_threshold: float = 0.2
    drift_metrics_port: int = 9101
    # Served predictions are buffered and bulk-inserted into inference_log.
    inference_log_flush_seconds: float = 1.0
    inference_log_buffer_size: int = 10_000

    # Comma-separated list of allowed CORS origins ("*" for any).
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
