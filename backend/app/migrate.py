"""
Apply database migrations.

    python -m app.migrate            # upgrade to head (what the compose "migrate" service runs)
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def alembic_config(database_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


def upgrade(database_url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(database_url), revision)


def downgrade(database_url: str, revision: str = "base") -> None:
    command.downgrade(alembic_config(database_url), revision)


def main() -> None:  # pragma: no cover - CLI wrapper
    from app.config import get_settings
    from app.observability import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    upgrade(settings.database_url)


if __name__ == "__main__":
    main()
