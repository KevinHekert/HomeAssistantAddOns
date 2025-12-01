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

    def test_fast_forward_with_existing_samples_and_gap(self, patch_db_engine):
        """Fast-forward should also work when samples exist but there's a gap > 24h.
        
        This tests the scenario from issue: importing a sensor halts when the gap
        between the last value and now is more than 1 day. The fix should fast-forward
        through the gap by checking subsequent 24-hour windows.
        """
        now_utc = datetime.now(timezone.utc)
        
        # Insert a sample from 5 days ago to simulate a gap
        sample_ts = now_utc - timedelta(days=5)
        with Session(patch_db_engine) as session:
            sample = Sample(
                entity_id="sensor.test",
                timestamp=sample_ts,
                value=10.0,
                unit="m/s",
            )
            session.add(sample)
            session.commit()

        call_count = [0]
        
        def mock_sync(entity_id, since):
            """
            Simulate sync returning 0 for the first few calls (gap period)
            then return 5 (data found) when we catch up.
            """
            call_count[0] += 1
            # After 3 iterations, pretend we found data
            if call_count[0] >= 3:
                return 5
            return 0

        # We need to track what sync_status is updated to after each sync
        # Since sync_history_for_entity updates sync_status, we need to track
        # the progression
        sync_timestamps = [sample_ts]  # Start with the sample timestamp
        
        def mock_get_sync_status(entity_id):
            """Return progressively newer timestamps as sync progresses."""
            status = MagicMock()
            # Each iteration moves forward by 1 day
            days_forward = call_count[0]  # 0, 1, 2, etc.
            status.last_attempt = sample_ts + timedelta(days=days_forward + 1)
            return status

        with patch.object(sensors_module, "sync_history_for_entity", side_effect=mock_sync), \
             patch.object(sensors_module, "get_sync_status", side_effect=mock_get_sync_status), \
             patch.object(sensors_module.time, "sleep") as mock_sleep:
            
            sensors_module._sync_entity("sensor.test")
            
            # Should call sync 3 times: 2 fast-forward iterations, then data found
            assert call_count[0] == 3
            # Sleep should be called once at the end
            mock_sleep.assert_called_once_with(5)

    def test_fast_forward_with_gap_stops_at_yesterday(self, patch_db_engine):
        """When samples exist with a gap, fast-forward should stop at yesterday.
        
        Even if no new data is found, the fast-forward should stop once we've
        caught up to yesterday to avoid infinite loops.
        """
        now_utc = datetime.now(timezone.utc)
        yesterday = now_utc - timedelta(days=1)
        
        # Insert a sample from 3 days ago to simulate a gap
        sample_ts = now_utc - timedelta(days=3)
        with Session(patch_db_engine) as session:
            sample = Sample(
                entity_id="sensor.test",
                timestamp=sample_ts,
                value=10.0,
                unit="m/s",
            )
            session.add(sample)
            session.commit()

        call_count = [0]
        
        def mock_sync(entity_id, since):
            """Always return 0 to simulate no data found in the gap."""
            call_count[0] += 1
            return 0

        def mock_get_sync_status(entity_id):
            """Return progressively newer timestamps as sync progresses."""
            status = MagicMock()
            # Each iteration moves forward by 1 day
            days_forward = call_count[0]
            new_ts = sample_ts + timedelta(days=days_forward + 1)
            # Cap at now to simulate reaching the present
            if new_ts > now_utc:
                new_ts = now_utc
            status.last_attempt = new_ts
            return status

        with patch.object(sensors_module, "sync_history_for_entity", side_effect=mock_sync), \
             patch.object(sensors_module, "get_sync_status", side_effect=mock_get_sync_status), \
             patch.object(sensors_module.time, "sleep") as mock_sleep:
            
            sensors_module._sync_entity("sensor.test")
            
            # Should stop after catching up to yesterday (about 2-3 iterations depending on timing)
            # The key assertion is that it stops without running max_iterations
            assert call_count[0] < 10  # Should stop well before max iterations
            assert call_count[0] >= 2  # Should have fast-forwarded at least once
            mock_sleep.assert_called_once_with(5)

    def test_dwh_sensor_large_gap_scenario(self, patch_db_engine):
        """Test the DWH sensor scenario from the issue: large gaps in history data.
        
        This simulates the case where a sensor (like DWH history) has data,
        then a gap of multiple days, and then more data. The sync should:
        1. Start from the last known sample
        2. Fast-forward through the gap (no data in those windows)
        3. Eventually find new data or catch up to yesterday
        """
        now_utc = datetime.now(timezone.utc)
        
        # Simulate DWH sensor with:
        # - Data from 10 days ago
        # - Gap of 7 days (no data)
        # - New data starts 3 days ago
        sample_ts = now_utc - timedelta(days=10)
        with Session(patch_db_engine) as session:
            sample = Sample(
                entity_id="sensor.dwh_history",
                timestamp=sample_ts,
                value=50.5,
                unit="°C",
            )
            session.add(sample)
            session.commit()

        call_count = [0]
        
        def mock_sync(entity_id, since):
            """
            Simulate the DWH sensor sync:
            - First 7 calls (days 10-4): no data (gap period)
            - After that: data found
            """
            call_count[0] += 1
            # Calculate which day we're syncing
            # since should progress by ~1 day each call after the first
            # Gap period is days 10 to 4, so about 6-7 iterations with no data
            if call_count[0] <= 6:
                return 0  # No data in this window (gap period)
            return 5  # Found data after the gap

        def mock_get_sync_status(entity_id):
            """Return progressively newer timestamps as sync progresses."""
            status = MagicMock()
            # Start from sample_ts, each iteration advances by ~1 day
            days_forward = call_count[0]
            status.last_attempt = sample_ts + timedelta(days=days_forward + 1)
            return status

        with patch.object(sensors_module, "sync_history_for_entity", side_effect=mock_sync), \
             patch.object(sensors_module, "get_sync_status", side_effect=mock_get_sync_status), \
             patch.object(sensors_module.time, "sleep") as mock_sleep:
            
            sensors_module._sync_entity("sensor.dwh_history")
            
            # Should fast-forward through the gap and find data
            # 6 iterations with no data (gap) + 1 iteration with data = 7 total
            assert call_count[0] == 7
            # Should complete with a single sleep at the end
            mock_sleep.assert_called_once_with(5)

    def test_dwh_sensor_gap_with_no_new_data_ever(self, patch_db_engine):
        """Test DWH sensor scenario where no new data is ever found after the gap.
        
        This tests the case where the sensor had data but is now inactive.
        The sync should still fast-forward through and stop at yesterday,
        not hang forever.
        """
        now_utc = datetime.now(timezone.utc)
        yesterday = now_utc - timedelta(days=1)
        
        # Simulate DWH sensor with old data from 5 days ago
        sample_ts = now_utc - timedelta(days=5)
        with Session(patch_db_engine) as session:
            sample = Sample(
                entity_id="sensor.dwh_inactive",
                timestamp=sample_ts,
                value=45.0,
                unit="°C",
            )
            session.add(sample)
            session.commit()

        call_count = [0]
        
        def mock_sync(entity_id, since):
            """Always return 0 - no new data ever found."""
            call_count[0] += 1
            return 0

        def mock_get_sync_status(entity_id):
            """Return progressively newer timestamps as sync progresses."""
            status = MagicMock()
            days_forward = call_count[0]
            new_ts = sample_ts + timedelta(days=days_forward + 1)
            # Cap at now to prevent going into the future
            if new_ts > now_utc:
                new_ts = now_utc
            status.last_attempt = new_ts
            return status

        with patch.object(sensors_module, "sync_history_for_entity", side_effect=mock_sync), \
             patch.object(sensors_module, "get_sync_status", side_effect=mock_get_sync_status), \
             patch.object(sensors_module.time, "sleep") as mock_sleep:
            
            sensors_module._sync_entity("sensor.dwh_inactive")
            
            # Should fast-forward through the gap and stop at yesterday
            # Gap is 5 days - 1 (yesterday) = 4 days worth of fast-forward
            # Should complete in < 10 iterations
            assert call_count[0] < 10
            assert call_count[0] >= 4  # At least covered the gap
            mock_sleep.assert_called_once_with(5)
