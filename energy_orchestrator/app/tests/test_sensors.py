"""
Tests for the sensors worker module.

Tests the sync entity functionality including fast-forward behavior
when no historic data is found.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import Base, Sample, SyncStatus
import workers.sensors as sensors_module


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def patch_db_engine(test_engine, monkeypatch):
    """Patch the engine in relevant modules."""
    import db.samples as samples_mod
    import db.sync_state as sync_state_mod

    monkeypatch.setattr(samples_mod, "engine", test_engine)
    monkeypatch.setattr(sync_state_mod, "engine", test_engine)
    return test_engine


class TestSyncEntity:
    """Test the _sync_entity function."""

    def test_empty_entity_id_returns_immediately(self):
        """Empty entity_id should return without doing anything."""
        with patch.object(sensors_module, "sync_history_for_entity") as mock_sync:
            sensors_module._sync_entity("")
            mock_sync.assert_not_called()

        with patch.object(sensors_module, "sync_history_for_entity") as mock_sync:
            sensors_module._sync_entity(None)
            mock_sync.assert_not_called()

    def test_normal_sync_when_data_found(self, patch_db_engine):
        """Normal sync with 5-second delay when data is found."""
        with patch.object(sensors_module, "sync_history_for_entity", return_value=5) as mock_sync, \
             patch.object(sensors_module.time, "sleep") as mock_sleep:
            
            sensors_module._sync_entity("sensor.test")
            
            # Should call sync once and sleep once
            assert mock_sync.call_count == 1
            mock_sleep.assert_called_once_with(5)

    def test_normal_sync_when_samples_exist(self, patch_db_engine):
        """Normal sync when entity already has samples in DB."""
        # Insert a sample so latest_ts is not None
        with Session(patch_db_engine) as session:
            sample = Sample(
                entity_id="sensor.test",
                timestamp=datetime.now(timezone.utc),
                value=10.0,
                unit="m/s",
            )
            session.add(sample)
            session.commit()

        with patch.object(sensors_module, "sync_history_for_entity", return_value=0) as mock_sync, \
             patch.object(sensors_module.time, "sleep") as mock_sleep:
            
            sensors_module._sync_entity("sensor.test")
            
            # Should call sync once and sleep once (no fast-forward when samples exist)
            assert mock_sync.call_count == 1
            mock_sleep.assert_called_once_with(5)

    def test_fast_forward_when_no_data_and_before_yesterday(self, patch_db_engine):
        """Fast-forward loop when no data found and not caught up to yesterday."""
        # Simulate: first call returns 0 with last_attempt from 10 days ago,
        # second call returns 0 with last_attempt from 5 days ago (still before yesterday),
        # third call returns 0 but last_attempt is now after yesterday (stop fast-forward)

        now_utc = datetime.now(timezone.utc)
        
        call_count = [0]
        
        def mock_get_sync_status(entity_id):
            """Return progressively newer timestamps."""
            call_count[0] += 1
            status = MagicMock()
            if call_count[0] == 1:
                # First call: 10 days ago
                status.last_attempt = now_utc - timedelta(days=10)
            elif call_count[0] == 2:
                # Second call: 5 days ago (still before yesterday)
                status.last_attempt = now_utc - timedelta(days=5)
            else:
                # Third call: today (after yesterday)
                status.last_attempt = now_utc - timedelta(hours=12)
            return status

        with patch.object(sensors_module, "sync_history_for_entity", return_value=0) as mock_sync, \
             patch.object(sensors_module, "get_sync_status", side_effect=mock_get_sync_status), \
             patch.object(sensors_module, "get_latest_sample_timestamp", return_value=None), \
             patch.object(sensors_module.time, "sleep") as mock_sleep:
            
            sensors_module._sync_entity("sensor.test")
            
            # Should call sync 3 times (fast-forward twice, then stop)
            assert mock_sync.call_count == 3
            # Sleep should only be called once (after final sync)
            mock_sleep.assert_called_once_with(5)

    def test_fast_forward_stops_when_data_found(self, patch_db_engine):
        """Fast-forward stops immediately when data is found."""
        call_count = [0]
        
        def mock_sync(entity_id, since):
            """Return 0 first time, then some data second time."""
            call_count[0] += 1
            if call_count[0] == 1:
                return 0
            return 5  # Data found

        now_utc = datetime.now(timezone.utc)
        
        sync_status_calls = [0]
        
        def mock_get_sync_status(entity_id):
            """Return timestamps before yesterday."""
            sync_status_calls[0] += 1
            status = MagicMock()
            status.last_attempt = now_utc - timedelta(days=10 - sync_status_calls[0])
            return status

        with patch.object(sensors_module, "sync_history_for_entity", side_effect=mock_sync), \
             patch.object(sensors_module, "get_sync_status", side_effect=mock_get_sync_status), \
             patch.object(sensors_module, "get_latest_sample_timestamp", return_value=None), \
             patch.object(sensors_module.time, "sleep") as mock_sleep:
            
            sensors_module._sync_entity("sensor.test")
            
            # Should stop after second call because data was found
            assert call_count[0] == 2
            mock_sleep.assert_called_once_with(5)

    def test_fast_forward_stops_at_yesterday(self, patch_db_engine):
        """Fast-forward stops when we reach yesterday."""
        now_utc = datetime.now(timezone.utc)
        yesterday = now_utc - timedelta(days=1)
        
        call_count = [0]
        
        def mock_get_sync_status(entity_id):
            """First call: before yesterday, second call: after yesterday."""
            call_count[0] += 1
            status = MagicMock()
            if call_count[0] == 1:
                status.last_attempt = now_utc - timedelta(days=5)
            else:
                status.last_attempt = yesterday + timedelta(hours=1)
            return status

        with patch.object(sensors_module, "sync_history_for_entity", return_value=0), \
             patch.object(sensors_module, "get_sync_status", side_effect=mock_get_sync_status), \
             patch.object(sensors_module, "get_latest_sample_timestamp", return_value=None), \
             patch.object(sensors_module.time, "sleep") as mock_sleep:
            
            sensors_module._sync_entity("sensor.test")
            
            # Should call sync twice (fast-forward once, then stop at yesterday)
            assert call_count[0] == 2
            mock_sleep.assert_called_once_with(5)

    def test_no_fast_forward_when_effective_since_is_none_first_iteration(self, patch_db_engine):
        """Test behavior when effective_since starts as None (first sync ever)."""
        now_utc = datetime.now(timezone.utc)
        
        call_count = [0]
        
        def mock_get_sync_status(entity_id):
            """First call: None, subsequent calls: progressively newer."""
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # No sync status yet
            status = MagicMock()
            if call_count[0] == 2:
                # After first sync, last_attempt is set to backfill start + 1 day
                status.last_attempt = now_utc - timedelta(days=99)
            else:
                # After more syncs, caught up to yesterday
                status.last_attempt = now_utc - timedelta(hours=12)
            return status

        with patch.object(sensors_module, "sync_history_for_entity", return_value=0), \
             patch.object(sensors_module, "get_sync_status", side_effect=mock_get_sync_status), \
             patch.object(sensors_module, "get_latest_sample_timestamp", return_value=None), \
             patch.object(sensors_module.time, "sleep") as mock_sleep:
            
            sensors_module._sync_entity("sensor.test")
            
            # Should fast-forward: None -> 99 days ago -> caught up
            assert call_count[0] == 3
            mock_sleep.assert_called_once_with(5)

    def test_max_iterations_limit(self, patch_db_engine):
        """Test that the safety limit prevents infinite loops."""
        now_utc = datetime.now(timezone.utc)
        
        call_count = [0]
        
        def mock_get_sync_status(entity_id):
            """Always return a timestamp before yesterday to simulate infinite loop scenario."""
            call_count[0] += 1
            status = MagicMock()
            # Always return a date far in the past
            status.last_attempt = now_utc - timedelta(days=50)
            return status

        with patch.object(sensors_module, "sync_history_for_entity", return_value=0), \
             patch.object(sensors_module, "get_sync_status", side_effect=mock_get_sync_status), \
             patch.object(sensors_module, "get_latest_sample_timestamp", return_value=None), \
             patch.object(sensors_module.time, "sleep") as mock_sleep:
            
            sensors_module._sync_entity("sensor.test")
            
            # Should stop at max_iterations (200) and sleep
            assert call_count[0] == 200
            mock_sleep.assert_called_once_with(5)
