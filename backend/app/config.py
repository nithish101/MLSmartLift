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

    # Comma-separated list of allowed CORS origins ("*" for any).
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
