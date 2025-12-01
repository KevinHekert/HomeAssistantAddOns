"""
Tests for the samples module.

Tests timestamp alignment to 5-second boundaries and upsert logic.
"""

import pytest
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import Base, Sample
from db.samples import (
    _align_timestamp_to_5s,
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


class TestAlignTimestampTo5s:
    """Test the _align_timestamp_to_5s function."""

    def test_already_aligned(self):
        """Timestamp already on 5-second boundary stays the same."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        assert _align_timestamp_to_5s(dt) == datetime(2024, 1, 1, 12, 0, 0)

        dt = datetime(2024, 1, 1, 12, 0, 5)
        assert _align_timestamp_to_5s(dt) == datetime(2024, 1, 1, 12, 0, 5)

        dt = datetime(2024, 1, 1, 12, 0, 55)
        assert _align_timestamp_to_5s(dt) == datetime(2024, 1, 1, 12, 0, 55)

    def test_rounds_down(self):
        """Timestamp not on boundary rounds down to nearest 5 seconds."""
        dt = datetime(2024, 1, 1, 12, 0, 2)
        assert _align_timestamp_to_5s(dt) == datetime(2024, 1, 1, 12, 0, 0)

        dt = datetime(2024, 1, 1, 12, 0, 7)
        assert _align_timestamp_to_5s(dt) == datetime(2024, 1, 1, 12, 0, 5)

        dt = datetime(2024, 1, 1, 12, 0, 59)
        assert _align_timestamp_to_5s(dt) == datetime(2024, 1, 1, 12, 0, 55)

    def test_strips_microseconds(self):
        """Microseconds are stripped."""
        dt = datetime(2024, 1, 1, 12, 0, 0, 123456)
        assert _align_timestamp_to_5s(dt) == datetime(2024, 1, 1, 12, 0, 0, 0)

        dt = datetime(2024, 1, 1, 12, 0, 3, 999999)
        assert _align_timestamp_to_5s(dt) == datetime(2024, 1, 1, 12, 0, 0, 0)


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

    def test_timestamp_aligned_on_insert(self, patch_engine):
        """Timestamp is aligned to 5-second boundary on insert."""
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 3),  # Not aligned
            10.0,
            "m/s",
        )

        with Session(patch_engine) as session:
            samples = session.query(Sample).all()
            assert len(samples) == 1
            assert samples[0].timestamp == datetime(2024, 1, 1, 12, 0, 0)  # Aligned

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

    def test_upsert_with_aligned_timestamps(self, patch_engine):
        """Different raw timestamps that align to same boundary result in one record."""
        # Insert sample at 12:00:02
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 2),
            10.0,
            "m/s",
        )

        # Insert sample at 12:00:03 - aligns to same boundary (12:00:00)
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 3),
            20.0,
            "m/s",
        )

        with Session(patch_engine) as session:
            samples = session.query(Sample).all()
            # Should only have 1 sample
            assert len(samples) == 1
            # Both aligned to 12:00:00
            assert samples[0].timestamp == datetime(2024, 1, 1, 12, 0, 0)
            # Value should be the latest (20.0)
            assert samples[0].value == 20.0

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

    def test_sample_exists_with_aligned_timestamp(self, patch_engine):
        """Works with timestamps that align to same boundary."""
        log_sample(
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            10.0,
            "m/s",
        )

        # Check with unaligned timestamp that aligns to 12:00:00
        assert sample_exists("sensor.test", datetime(2024, 1, 1, 12, 0, 2)) is True


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
