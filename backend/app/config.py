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

    # Path to the fitted scikit-learn pipeline produced by `python -m app.ml.train`.
    model_path: str = "artifacts/model.joblib"

    # Comma-separated list of allowed CORS origins ("*" for any).
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
