"""The Alembic migrations build exactly the schema the models describe, and reverse cleanly."""

import uuid

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from app.db import Base
from app.migrate import downgrade, upgrade

pytestmark = pytest.mark.integration


@pytest.fixture
def fresh_db(database_url):
    """An empty database on the same server, dropped afterwards."""
    name = f"mig_{uuid.uuid4().hex[:8]}"
    admin = create_engine(database_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(database_url).set(database=name).render_as_string(hide_password=False)
    yield url
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    admin.dispose()


def schema_diff(url):
    engine = create_engine(url)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)
    engine.dispose()
    return diff


def tables(url):
    engine = create_engine(url)
    names = set(inspect(engine).get_table_names())
    engine.dispose()
    return names


def test_upgrade_matches_models_and_round_trips(fresh_db):
    upgrade(fresh_db)
    expected = set(Base.metadata.tables)
    assert tables(fresh_db) == expected | {"alembic_version"}
    assert schema_diff(fresh_db) == []  # same check as `alembic check`

    downgrade(fresh_db, "base")
    assert tables(fresh_db) == {"alembic_version"}

    upgrade(fresh_db)
    assert schema_diff(fresh_db) == []
