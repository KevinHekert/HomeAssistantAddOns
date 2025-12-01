"""
Tests for the sync_state module.

Tests the sync status tracking functionality for sensor data synchronization.
"""

import pytest
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import Base, SyncStatus
from db.sync_state import (
    update_sync_attempt,
    get_sync_status,
)
import db.core as core_module
import db.sync_state as sync_state_module


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def patch_engine(test_engine, monkeypatch):
    """Patch the engine in relevant modules."""
    monkeypatch.setattr(core_module, "engine", test_engine)
    monkeypatch.setattr(sync_state_module, "engine", test_engine)
    return test_engine


class TestUpdateSyncAttempt:
    """Test the update_sync_attempt function."""

    def test_creates_new_status_on_first_attempt(self, patch_engine):
        """Creates a new SyncStatus record for first sync attempt."""
        entity_id = "sensor.test"
        attempt_ts = datetime(2024, 1, 1, 12, 0, 0)

        update_sync_attempt(entity_id, attempt_ts, success=False)

        with Session(patch_engine) as session:
            status = session.get(SyncStatus, entity_id)
            assert status is not None
            assert status.entity_id == entity_id
            assert status.last_attempt == attempt_ts
            assert status.last_success is None

    def test_updates_last_success_when_successful(self, patch_engine):
        """Updates last_success when success=True."""
        entity_id = "sensor.test"
        attempt_ts = datetime(2024, 1, 1, 12, 0, 0)

        update_sync_attempt(entity_id, attempt_ts, success=True)

        with Session(patch_engine) as session:
            status = session.get(SyncStatus, entity_id)
            assert status is not None
            assert status.last_attempt == attempt_ts
            assert status.last_success == attempt_ts

    def test_updates_existing_status(self, patch_engine):
        """Updates an existing SyncStatus record."""
        entity_id = "sensor.test"
        first_attempt = datetime(2024, 1, 1, 12, 0, 0)
        second_attempt = datetime(2024, 1, 1, 13, 0, 0)

        # First attempt (failed)
        update_sync_attempt(entity_id, first_attempt, success=False)

        # Second attempt (successful)
        update_sync_attempt(entity_id, second_attempt, success=True)

        with Session(patch_engine) as session:
            status = session.get(SyncStatus, entity_id)
            assert status is not None
            assert status.last_attempt == second_attempt
            assert status.last_success == second_attempt

    def test_preserves_last_success_on_failed_attempt(self, patch_engine):
        """last_success is preserved when a subsequent attempt fails."""
        entity_id = "sensor.test"
        first_attempt = datetime(2024, 1, 1, 12, 0, 0)
        second_attempt = datetime(2024, 1, 1, 13, 0, 0)

        # First attempt (successful)
        update_sync_attempt(entity_id, first_attempt, success=True)

        # Second attempt (failed)
        update_sync_attempt(entity_id, second_attempt, success=False)

        with Session(patch_engine) as session:
            status = session.get(SyncStatus, entity_id)
            assert status is not None
            assert status.last_attempt == second_attempt
            assert status.last_success == first_attempt  # Preserved

    def test_handles_multiple_entities(self, patch_engine):
        """Handles multiple entities independently."""
        entity1 = "sensor.test1"
        entity2 = "sensor.test2"
        ts1 = datetime(2024, 1, 1, 12, 0, 0)
        ts2 = datetime(2024, 1, 1, 13, 0, 0)

        update_sync_attempt(entity1, ts1, success=True)
        update_sync_attempt(entity2, ts2, success=False)

        with Session(patch_engine) as session:
            status1 = session.get(SyncStatus, entity1)
            status2 = session.get(SyncStatus, entity2)

            assert status1.last_attempt == ts1
            assert status1.last_success == ts1
            assert status2.last_attempt == ts2
            assert status2.last_success is None


class TestGetSyncStatus:
    """Test the get_sync_status function."""

    def test_returns_none_when_no_status_exists(self, patch_engine):
        """Returns None when no status exists for entity."""
        result = get_sync_status("sensor.nonexistent")
        assert result is None

    def test_returns_status_when_exists(self, patch_engine):
        """Returns SyncStatus when it exists."""
        entity_id = "sensor.test"
        attempt_ts = datetime(2024, 1, 1, 12, 0, 0)

        update_sync_attempt(entity_id, attempt_ts, success=True)

        result = get_sync_status(entity_id)
        assert result is not None
        assert result.entity_id == entity_id
        assert result.last_attempt == attempt_ts
        assert result.last_success == attempt_ts

    def test_returns_correct_status_for_entity(self, patch_engine):
        """Returns the correct status for a specific entity."""
        entity1 = "sensor.test1"
        entity2 = "sensor.test2"
        ts1 = datetime(2024, 1, 1, 12, 0, 0)
        ts2 = datetime(2024, 1, 1, 13, 0, 0)

        update_sync_attempt(entity1, ts1, success=True)
        update_sync_attempt(entity2, ts2, success=False)

        result = get_sync_status(entity1)
        assert result.entity_id == entity1
        assert result.last_attempt == ts1

        result = get_sync_status(entity2)
        assert result.entity_id == entity2
        assert result.last_attempt == ts2
