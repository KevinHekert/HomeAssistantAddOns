"""
Tests for the samples module.

Tests timestamp normalization and upsert logic.
"""

import pytest
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import Base, Sample
from db.samples import (
    _normalize_timestamp,
    get_latest_sample_timestamp,
    log_sample,
    sample_exists,
)
import db.samples as samples_module


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def patch_engine(test_engine, monkeypatch):
    """Patch the engine in samples module."""
    monkeypatch.setattr(samples_module, "engine", test_engine)
    return test_engine


class TestNormalizeTimestamp:
    """Test the _normalize_timestamp function."""

    def test_preserves_seconds(self):
        """Timestamp seconds are preserved (not rounded)."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        assert _normalize_timestamp(dt) == datetime(2024, 1, 1, 12, 0, 0)

        dt = datetime(2024, 1, 1, 12, 0, 7)
        assert _normalize_timestamp(dt) == datetime(2024, 1, 1, 12, 0, 7)

        dt = datetime(2024, 1, 1, 12, 0, 59)
        assert _normalize_timestamp(dt) == datetime(2024, 1, 1, 12, 0, 59)

    def test_strips_microseconds(self):
        """Microseconds are stripped."""
        dt = datetime(2024, 1, 1, 12, 0, 0, 123456)
        assert _normalize_timestamp(dt) == datetime(2024, 1, 1, 12, 0, 0, 0)

        dt = datetime(2024, 1, 1, 12, 0, 3, 999999)
        assert _normalize_timestamp(dt) == datetime(2024, 1, 1, 12, 0, 3, 0)


class TestLogSample:
    """Test the log_sample function with upsert behavior."""

    def test_insert_new_sample(self, patch_engine):
        """New sample is inserted correctly."""
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            10.0,
            "m/s",
        )

        with Session(patch_engine) as session:
            samples = session.query(Sample).all()
            assert len(samples) == 1
            assert samples[0].entity_id == "sensor.test"
            assert samples[0].timestamp == datetime(2024, 1, 1, 12, 0, 0)
            assert samples[0].value == 10.0
            assert samples[0].unit == "m/s"

    def test_timestamp_microseconds_stripped_on_insert(self, patch_engine):
        """Timestamp microseconds are stripped on insert, but seconds preserved."""
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 3, 123456),  # With microseconds
            10.0,
            "m/s",
        )

        with Session(patch_engine) as session:
            samples = session.query(Sample).all()
            assert len(samples) == 1
            assert samples[0].timestamp == datetime(2024, 1, 1, 12, 0, 3)  # Microseconds stripped, seconds preserved

    def test_upsert_updates_existing(self, patch_engine):
        """If sample exists, update instead of creating duplicate."""
        # Insert first sample
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            10.0,
            "m/s",
        )

        # Insert same entity + timestamp with different value
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            20.0,
            "m/s",
        )

        with Session(patch_engine) as session:
            samples = session.query(Sample).all()
            # Should only have 1 sample, not 2
            assert len(samples) == 1
            # Value should be updated
            assert samples[0].value == 20.0

    def test_upsert_with_same_timestamps(self, patch_engine):
        """Different raw timestamps with same normalized timestamp (only microseconds differ) result in one record."""
        # Insert sample at 12:00:02.000
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 2, 0),
            10.0,
            "m/s",
        )

        # Insert sample at 12:00:02.500 - normalizes to same (12:00:02)
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 2, 500000),
            20.0,
            "m/s",
        )

        with Session(patch_engine) as session:
            samples = session.query(Sample).all()
            # Should only have 1 sample
            assert len(samples) == 1
            # Normalized to 12:00:02
            assert samples[0].timestamp == datetime(2024, 1, 1, 12, 0, 2)
            # Value should be the latest (20.0)
            assert samples[0].value == 20.0

    def test_different_timestamps_not_merged(self, patch_engine):
        """Different timestamps (even 1 second apart) are stored separately."""
        # Insert sample at 12:00:02
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 2),
            10.0,
            "m/s",
        )

        # Insert sample at 12:00:03 - different second, stored separately
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 3),
            20.0,
            "m/s",
        )

        with Session(patch_engine) as session:
            samples = session.query(Sample).order_by(Sample.timestamp).all()
            # Should have 2 separate samples
            assert len(samples) == 2
            assert samples[0].timestamp == datetime(2024, 1, 1, 12, 0, 2)
            assert samples[0].value == 10.0
            assert samples[1].timestamp == datetime(2024, 1, 1, 12, 0, 3)
            assert samples[1].value == 20.0

    def test_different_entities_not_merged(self, patch_engine):
        """Different entities at same timestamp are not merged."""
        log_sample(
            "sensor.test1",
            datetime(2024, 1, 1, 12, 0, 0),
            10.0,
            "m/s",
        )

        log_sample(
            "sensor.test2",
            datetime(2024, 1, 1, 12, 0, 0),
            20.0,
            "m/s",
        )

        with Session(patch_engine) as session:
            samples = session.query(Sample).all()
            assert len(samples) == 2

    def test_null_value_skipped(self, patch_engine):
        """None value is not inserted."""
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            None,
            "m/s",
        )

        with Session(patch_engine) as session:
            samples = session.query(Sample).all()
            assert len(samples) == 0


class TestSampleExists:
    """Test the sample_exists function."""

    def test_sample_does_not_exist(self, patch_engine):
        """Returns False when sample does not exist."""
        assert sample_exists("sensor.test", datetime(2024, 1, 1, 12, 0, 0)) is False

    def test_sample_exists(self, patch_engine):
        """Returns True when sample exists."""
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            10.0,
            "m/s",
        )

        assert sample_exists("sensor.test", datetime(2024, 1, 1, 12, 0, 0)) is True

    def test_sample_exists_with_normalized_timestamp(self, patch_engine):
        """Works with timestamps that normalize to same (microseconds stripped)."""
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 2),
            10.0,
            "m/s",
        )

        # Check with timestamp that has microseconds (normalizes to 12:00:02)
        assert sample_exists("sensor.test", datetime(2024, 1, 1, 12, 0, 2, 123456)) is True
        
    def test_sample_exists_different_seconds(self, patch_engine):
        """Different seconds return False."""
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 2),
            10.0,
            "m/s",
        )

        # Check with different second (should not exist)
        assert sample_exists("sensor.test", datetime(2024, 1, 1, 12, 0, 3)) is False


class TestGetLatestSampleTimestamp:
    """Test the get_latest_sample_timestamp function."""

    def test_no_samples(self, patch_engine):
        """Returns None when no samples exist."""
        result = get_latest_sample_timestamp("sensor.test")
        assert result is None

    def test_single_sample(self, patch_engine):
        """Returns timestamp of single sample."""
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            10.0,
            "m/s",
        )

        result = get_latest_sample_timestamp("sensor.test")
        assert result == datetime(2024, 1, 1, 12, 0, 0)

    def test_multiple_samples(self, patch_engine):
        """Returns latest timestamp from multiple samples."""
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            10.0,
            "m/s",
        )
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 13, 0, 0),
            15.0,
            "m/s",
        )
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 30, 0),
            12.0,
            "m/s",
        )

        result = get_latest_sample_timestamp("sensor.test")
        assert result == datetime(2024, 1, 1, 13, 0, 0)


class TestGetSensorInfo:
    """Test the get_sensor_info function."""

    def test_no_samples(self, patch_engine):
        """Returns empty list when no samples exist."""
        from db.samples import get_sensor_info
        result = get_sensor_info()
        assert result == []

    def test_single_sensor(self, patch_engine):
        """Returns info for single sensor."""
        from db.samples import get_sensor_info
        
        log_sample("sensor.test", datetime(2024, 1, 1, 12, 0, 0), 10.0, "m/s")
        log_sample("sensor.test", datetime(2024, 1, 1, 13, 0, 0), 15.0, "m/s")
        
        result = get_sensor_info()
        assert len(result) == 1
        assert result[0]["entity_id"] == "sensor.test"
        assert result[0]["first_timestamp"] == "2024-01-01T12:00:00"
        assert result[0]["last_timestamp"] == "2024-01-01T13:00:00"
        assert result[0]["sample_count"] == 2

    def test_multiple_sensors(self, patch_engine):
        """Returns info for multiple sensors."""
        from db.samples import get_sensor_info
        
        log_sample("sensor.a", datetime(2024, 1, 1, 10, 0, 0), 1.0, "°C")
        log_sample("sensor.a", datetime(2024, 1, 1, 12, 0, 0), 2.0, "°C")
        log_sample("sensor.b", datetime(2024, 1, 1, 11, 0, 0), 3.0, "m/s")
        
        result = get_sensor_info()
        assert len(result) == 2
        
        # Results should be ordered by entity_id
        sensor_a = next(s for s in result if s["entity_id"] == "sensor.a")
        sensor_b = next(s for s in result if s["entity_id"] == "sensor.b")
        
        assert sensor_a["sample_count"] == 2
        assert sensor_a["first_timestamp"] == "2024-01-01T10:00:00"
        assert sensor_a["last_timestamp"] == "2024-01-01T12:00:00"
        
        assert sensor_b["sample_count"] == 1
        assert sensor_b["first_timestamp"] == "2024-01-01T11:00:00"
        assert sensor_b["last_timestamp"] == "2024-01-01T11:00:00"


class TestGetResampledSensorInfo:
    """Test the get_resampled_sensor_info function."""

    def test_no_resampled_samples(self, patch_engine):
        """Returns empty list when no resampled samples exist."""
        from db.samples import get_resampled_sensor_info
        result = get_resampled_sensor_info()
        assert result == []

    def test_single_category(self, patch_engine):
        """Returns info for single category."""
        from db.samples import get_resampled_sensor_info
        from db import ResampledSample
        from sqlalchemy.orm import Session
        
        # Add some resampled samples
        with Session(patch_engine) as session:
            session.add(ResampledSample(
                slot_start=datetime(2024, 1, 1, 12, 0, 0),
                category="outdoor_temp",
                value=5.5,
                unit="°C",
                is_derived=False
            ))
            session.add(ResampledSample(
                slot_start=datetime(2024, 1, 1, 13, 0, 0),
                category="outdoor_temp",
                value=6.0,
                unit="°C",
                is_derived=False
            ))
            session.commit()
        
        result = get_resampled_sensor_info()
        assert len(result) == 1
        assert result[0]["category"] == "outdoor_temp"
        assert result[0]["unit"] == "°C"
        assert result[0]["is_derived"] is False
        assert result[0]["first_timestamp"] == "2024-01-01T12:00:00"
        assert result[0]["last_timestamp"] == "2024-01-01T13:00:00"
        assert result[0]["sample_count"] == 2

    def test_multiple_categories(self, patch_engine):
        """Returns info for multiple categories."""
        from db.samples import get_resampled_sensor_info
        from db import ResampledSample
        from sqlalchemy.orm import Session
        
        # Add samples for multiple categories
        with Session(patch_engine) as session:
            # outdoor_temp - raw
            session.add(ResampledSample(
                slot_start=datetime(2024, 1, 1, 10, 0, 0),
                category="outdoor_temp",
                value=5.5,
                unit="°C",
                is_derived=False
            ))
            session.add(ResampledSample(
                slot_start=datetime(2024, 1, 1, 12, 0, 0),
                category="outdoor_temp",
                value=6.0,
                unit="°C",
                is_derived=False
            ))
            # indoor_temp - raw
            session.add(ResampledSample(
                slot_start=datetime(2024, 1, 1, 11, 0, 0),
                category="indoor_temp",
                value=20.0,
                unit="°C",
                is_derived=False
            ))
            # outdoor_temp_avg_1h - derived
            session.add(ResampledSample(
                slot_start=datetime(2024, 1, 1, 12, 0, 0),
                category="outdoor_temp_avg_1h",
                value=5.8,
                unit="°C",
                is_derived=True
            ))
            session.commit()
        
        result = get_resampled_sensor_info()
        assert len(result) == 3
        
        # Results should be ordered by category
        outdoor_temp = next(s for s in result if s["category"] == "outdoor_temp")
        indoor_temp = next(s for s in result if s["category"] == "indoor_temp")
        outdoor_temp_avg = next(s for s in result if s["category"] == "outdoor_temp_avg_1h")
        
        assert outdoor_temp["sample_count"] == 2
        assert outdoor_temp["first_timestamp"] == "2024-01-01T10:00:00"
        assert outdoor_temp["last_timestamp"] == "2024-01-01T12:00:00"
        assert outdoor_temp["is_derived"] is False
        
        assert indoor_temp["sample_count"] == 1
        assert indoor_temp["first_timestamp"] == "2024-01-01T11:00:00"
        assert indoor_temp["last_timestamp"] == "2024-01-01T11:00:00"
        assert indoor_temp["is_derived"] is False
        
        assert outdoor_temp_avg["sample_count"] == 1
        assert outdoor_temp_avg["is_derived"] is True

