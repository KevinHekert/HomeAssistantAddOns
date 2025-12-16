import logging
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

import db.core as core


class DummySqlError(SQLAlchemyError):
    """Concrete SQLAlchemyError subclass for testing."""


@contextmanager
def _mock_connection(scalar_return=1):
    """Context manager yielding a mocked DB connection."""
    execute_result = MagicMock()
    execute_result.scalar.return_value = scalar_return

    connection = MagicMock()
    connection.execute.return_value = execute_result
    connection.commit = MagicMock()
    yield connection


@pytest.fixture
def mock_engine(monkeypatch):
    """Provide a mock SQLAlchemy engine with a connect context manager."""
    engine = MagicMock()
    with _mock_connection() as connection:
        managed_connection = connection

    @contextmanager
    def connect_ctx():
        yield managed_connection

    engine.connect = MagicMock(side_effect=connect_ctx)
    monkeypatch.setattr(core, "engine", engine)
    return engine, managed_connection


def test_test_db_connection_success(mock_engine):
    """test_db_connection executes a simple SELECT on the engine."""
    engine, connection = mock_engine
    core.test_db_connection()

    engine.connect.assert_called_once()
    connection.execute.assert_called_once()


def test_test_db_connection_logs_error(monkeypatch, caplog):
    """Errors during DB connection are logged without raising."""
    failing_engine = MagicMock()

    def connect_ctx():
        raise DummySqlError("boom")

    failing_engine.connect = MagicMock(side_effect=connect_ctx)
    monkeypatch.setattr(core, "engine", failing_engine)

    with caplog.at_level(logging.ERROR):
        core.test_db_connection()

    assert any("Fout bij verbinden" in msg for msg in caplog.messages)


@pytest.fixture
def mock_schema_components(monkeypatch):
    """Patch Base.metadata.create_all and migration helpers."""
    create_all = MagicMock()
    monkeypatch.setattr(core.Base.metadata, "create_all", create_all)

    migrations = {}
    for name in [
        "_migrate_add_is_derived_column",
        "_migrate_add_optimizer_row_data_columns",
        "_migrate_add_complete_feature_config_column",
    ]:
        migrations[name] = MagicMock()
        monkeypatch.setattr(core, name, migrations[name])
    return create_all, migrations


def test_init_db_schema_runs_migrations(mock_engine, mock_schema_components):
    """init_db_schema creates tables and runs migrations."""
    create_all, migrations = mock_schema_components

    core.init_db_schema()

    create_all.assert_called_once_with(core.engine)
    for migration in migrations.values():
        migration.assert_called_once()


def test_init_db_schema_logs_error(mock_schema_components, monkeypatch, caplog):
    """SQLAlchemy errors during init_db_schema are logged."""
    create_all, migrations = mock_schema_components
    create_all.side_effect = DummySqlError("schema fail")

    with caplog.at_level(logging.ERROR):
        core.init_db_schema()

    assert any("Fout bij aanmaken schema" in msg for msg in caplog.messages)
    for migration in migrations.values():
        migration.assert_not_called()


@pytest.fixture
def connection_factory(monkeypatch):
    """Allow tests to supply a mocked connection to core.engine.connect."""

    def set_connection(connection):
        engine = MagicMock()

        @contextmanager
        def connect_ctx():
            yield connection

        engine.connect = MagicMock(side_effect=connect_ctx)
        monkeypatch.setattr(core, "engine", engine)
        return engine

    return set_connection


def test_migrate_add_is_derived_adds_column(connection_factory):
    connection = MagicMock()
    connection.execute.return_value.scalar.return_value = 0
    connection.commit = MagicMock()

    engine = connection_factory(connection)

    core._migrate_add_is_derived_column()

    # First execute is the existence check; second is the ALTER statement
    assert connection.execute.call_count == 2
    connection.commit.assert_called_once()
    engine.connect.assert_called_once()


def test_migrate_add_is_derived_skips_when_exists(connection_factory):
    connection = MagicMock()
    connection.execute.return_value.scalar.return_value = 1

    engine = connection_factory(connection)

    core._migrate_add_is_derived_column()

    connection.execute.assert_called_once()  # Only the existence check
    connection.commit.assert_not_called()
    engine.connect.assert_called_once()


def test_migrate_add_optimizer_row_data_columns_adds_missing(connection_factory):
    connection = MagicMock()
    # First call: first_row_json missing (0). Second: last_row_json missing (0).
    connection.execute.return_value.scalar.side_effect = [0, 0]
    connection.commit = MagicMock()

    engine = connection_factory(connection)

    core._migrate_add_optimizer_row_data_columns()

    # Two existence checks + two ALTER statements
    assert connection.execute.call_count == 4
    assert connection.commit.call_count == 2
    engine.connect.assert_called_once()


def test_migrate_add_optimizer_row_data_columns_skips_existing(connection_factory):
    connection = MagicMock()
    connection.execute.return_value.scalar.side_effect = [1, 1]
    connection.commit = MagicMock()

    engine = connection_factory(connection)

    core._migrate_add_optimizer_row_data_columns()

    # Only the two existence checks should run
    assert connection.execute.call_count == 2
    connection.commit.assert_not_called()
    engine.connect.assert_called_once()


def test_migrate_add_complete_feature_config_column(connection_factory):
    connection = MagicMock()
    connection.execute.return_value.scalar.return_value = 0
    connection.commit = MagicMock()

    engine = connection_factory(connection)

    core._migrate_add_complete_feature_config_column()

    assert connection.execute.call_count == 2  # Check + alter
    connection.commit.assert_called_once()
    engine.connect.assert_called_once()


def test_migrate_add_complete_feature_config_column_skips_when_exists(connection_factory):
    connection = MagicMock()
    connection.execute.return_value.scalar.return_value = 1
    connection.commit = MagicMock()

    engine = connection_factory(connection)

    core._migrate_add_complete_feature_config_column()

    connection.execute.assert_called_once()
    connection.commit.assert_not_called()
    engine.connect.assert_called_once()
