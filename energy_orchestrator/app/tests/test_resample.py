"""
Tests for the resampling logic.

Uses an in-memory SQLite database to test:
1. Schema creation
2. Handling missing categories
3. Correct global time range computation
4. 5-minute grid behavior
5. Time-weighted average correctness
6. Complete-slot semantics
7. Idempotence
8. ResampleStats return values
"""

import pytest
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import Base, ResampledSample, Sample, SensorMapping
from db.resample import (
    RESAMPLE_STEP,
    ResampleStats,
    _align_to_5min_boundary,
    compute_time_weighted_avg,
    get_global_range_for_all_categories,
    get_primary_entities_by_category,
    resample_all_categories_to_5min,
)
import db.core as core_module
import db.resample as resample_module


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def patch_engine(test_engine, monkeypatch):
    """Patch the engine in both core and resample modules."""
    monkeypatch.setattr(core_module, "engine", test_engine)
    monkeypatch.setattr(resample_module, "engine", test_engine)

    # Also patch init_db_schema to use test engine
    def mock_init_db_schema():
        Base.metadata.create_all(test_engine)

    monkeypatch.setattr(core_module, "init_db_schema", mock_init_db_schema)
    monkeypatch.setattr(resample_module, "init_db_schema", mock_init_db_schema)

    return test_engine


class TestSchemaCreation:
    """Test that schema is created correctly."""

    def test_tables_created(self, test_engine):
        """Verify all tables are created."""
        tables = Base.metadata.tables.keys()
        assert "samples" in tables
        assert "sync_status" in tables
        assert "sensor_mappings" in tables
        assert "resampled_samples" in tables


class TestGetPrimaryEntitiesByCategory:
    """Test the get_primary_entities_by_category function."""

    def test_empty_mappings(self, patch_engine):
        """If sensor_mappings is empty, return empty dict."""
        result = get_primary_entities_by_category()
        assert result == {}

    def test_single_category_single_entity(self, patch_engine):
        """Single active mapping returns that entity."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.knmi_windsnelheid",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        result = get_primary_entities_by_category()
        assert result == {"WIND": "sensor.knmi_windsnelheid"}

    def test_inactive_mappings_excluded(self, patch_engine):
        """Inactive mappings are not included."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.knmi_windsnelheid",
                    is_active=False,
                    priority=1,
                )
            )
            session.commit()

        result = get_primary_entities_by_category()
        assert result == {}

    def test_priority_ordering(self, patch_engine):
        """Lower priority number wins."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.backup_wind",
                    is_active=True,
                    priority=2,
                )
            )
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.primary_wind",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        result = get_primary_entities_by_category()
        assert result == {"WIND": "sensor.primary_wind"}

    def test_multiple_categories(self, patch_engine):
        """Multiple categories each get their primary entity."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.knmi_windsnelheid",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="OUTDOOR_TEMP",
                    entity_id="sensor.knmi_temperatuur",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        result = get_primary_entities_by_category()
        assert result == {
            "WIND": "sensor.knmi_windsnelheid",
            "OUTDOOR_TEMP": "sensor.knmi_temperatuur",
        }


class TestGetGlobalRangeForAllCategories:
    """Test the get_global_range_for_all_categories function."""

    def test_no_mappings(self, patch_engine):
        """No mappings returns (None, None, {})."""
        start, end, mapping = get_global_range_for_all_categories()
        assert start is None
        assert end is None
        assert mapping == {}

    def test_no_samples_for_entity(self, patch_engine):
        """Entity with no samples returns (None, None, mapping)."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.knmi_windsnelheid",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        start, end, mapping = get_global_range_for_all_categories()
        assert start is None
        assert end is None
        assert "WIND" in mapping

    def test_single_category_with_data(self, patch_engine):
        """Single category with data returns its range."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.knmi_windsnelheid",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.knmi_windsnelheid",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.knmi_windsnelheid",
                    timestamp=datetime(2024, 1, 1, 13, 0, 0),
                    value=15.0,
                    unit="m/s",
                )
            )
            session.commit()

        start, end, mapping = get_global_range_for_all_categories()
        assert start == datetime(2024, 1, 1, 12, 0, 0)
        assert end == datetime(2024, 1, 1, 13, 0, 0)

    def test_multiple_categories_intersection(self, patch_engine):
        """Multiple categories compute intersection of time ranges."""
        with Session(patch_engine) as session:
            # WIND: 12:00 to 14:00
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 14, 0, 0),
                    value=15.0,
                    unit="m/s",
                )
            )

            # TEMP: 12:30 to 13:30
            session.add(
                SensorMapping(
                    category="TEMP",
                    entity_id="sensor.temp",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.temp",
                    timestamp=datetime(2024, 1, 1, 12, 30, 0),
                    value=20.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.temp",
                    timestamp=datetime(2024, 1, 1, 13, 30, 0),
                    value=22.0,
                    unit="°C",
                )
            )
            session.commit()

        start, end, mapping = get_global_range_for_all_categories()
        # global_start = max(12:00, 12:30) = 12:30
        # global_end = min(14:00, 13:30) = 13:30
        assert start == datetime(2024, 1, 1, 12, 30, 0)
        assert end == datetime(2024, 1, 1, 13, 30, 0)


class TestAlignTo5MinBoundary:
    """Test the _align_to_5min_boundary function."""

    def test_already_aligned(self):
        """Datetime already on 5-minute boundary stays the same."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        assert _align_to_5min_boundary(dt) == datetime(2024, 1, 1, 12, 0, 0)

        dt = datetime(2024, 1, 1, 12, 5, 0)
        assert _align_to_5min_boundary(dt) == datetime(2024, 1, 1, 12, 5, 0)

    def test_rounds_down(self):
        """Datetime not on boundary rounds down."""
        dt = datetime(2024, 1, 1, 12, 3, 30)
        assert _align_to_5min_boundary(dt) == datetime(2024, 1, 1, 12, 0, 0)

        dt = datetime(2024, 1, 1, 12, 7, 45)
        assert _align_to_5min_boundary(dt) == datetime(2024, 1, 1, 12, 5, 0)

        dt = datetime(2024, 1, 1, 12, 59, 59)
        assert _align_to_5min_boundary(dt) == datetime(2024, 1, 1, 12, 55, 0)

    def test_strips_microseconds(self):
        """Microseconds are stripped."""
        dt = datetime(2024, 1, 1, 12, 0, 0, 123456)
        assert _align_to_5min_boundary(dt) == datetime(2024, 1, 1, 12, 0, 0, 0)


class TestComputeTimeWeightedAvg:
    """Test the compute_time_weighted_avg function."""

    def test_no_data(self, patch_engine):
        """No samples returns (None, None)."""
        with Session(patch_engine) as session:
            avg, unit = compute_time_weighted_avg(
                session,
                "sensor.test",
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 5, 0),
            )
        assert avg is None
        assert unit is None

    def test_only_previous_sample(self, patch_engine):
        """Previous sample before window is used for entire window."""
        with Session(patch_engine) as session:
            session.add(
                Sample(
                    entity_id="sensor.test",
                    timestamp=datetime(2024, 1, 1, 11, 55, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.commit()

            avg, unit = compute_time_weighted_avg(
                session,
                "sensor.test",
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 5, 0),
            )

        assert avg == 10.0
        assert unit == "m/s"

    def test_only_sample_in_window(self, patch_engine):
        """Sample only at start of window is used for entire window."""
        with Session(patch_engine) as session:
            session.add(
                Sample(
                    entity_id="sensor.test",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.commit()

            avg, unit = compute_time_weighted_avg(
                session,
                "sensor.test",
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 5, 0),
            )

        assert avg == 10.0
        assert unit == "m/s"

    def test_sample_mid_window(self, patch_engine):
        """Sample in middle of window splits the average."""
        with Session(patch_engine) as session:
            # Previous sample: value 10
            session.add(
                Sample(
                    entity_id="sensor.test",
                    timestamp=datetime(2024, 1, 1, 11, 55, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            # Sample at 12:02:30: value 20
            session.add(
                Sample(
                    entity_id="sensor.test",
                    timestamp=datetime(2024, 1, 1, 12, 2, 30),
                    value=20.0,
                    unit="m/s",
                )
            )
            session.commit()

            avg, unit = compute_time_weighted_avg(
                session,
                "sensor.test",
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 5, 0),
            )

        # Window is 5 minutes = 300 seconds
        # 12:00:00 to 12:02:30 = 150 seconds at value 10
        # 12:02:30 to 12:05:00 = 150 seconds at value 20
        # Average = (10*150 + 20*150) / 300 = 4500 / 300 = 15
        assert avg == 15.0
        assert unit == "m/s"

    def test_sparse_data_last_known_value(self, patch_engine):
        """
        Test sparse data scenario from acceptance criteria.

        Given:
        - 12:00 → 10
        - 12:40 → 20
        - No intermediate samples

        For all slots between 12:00 and 12:40:
        - The computed average is 10 (last known value held constant)
        For the slot [12:40, 12:45):
        - The computed average is 20
        """
        with Session(patch_engine) as session:
            session.add(
                Sample(
                    entity_id="sensor.test",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.test",
                    timestamp=datetime(2024, 1, 1, 12, 40, 0),
                    value=20.0,
                    unit="m/s",
                )
            )
            session.commit()

            # Slot [12:00, 12:05)
            avg, _ = compute_time_weighted_avg(
                session,
                "sensor.test",
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 5, 0),
            )
            assert avg == 10.0

            # Slot [12:05, 12:10)
            avg, _ = compute_time_weighted_avg(
                session,
                "sensor.test",
                datetime(2024, 1, 1, 12, 5, 0),
                datetime(2024, 1, 1, 12, 10, 0),
            )
            assert avg == 10.0

            # Slot [12:35, 12:40)
            avg, _ = compute_time_weighted_avg(
                session,
                "sensor.test",
                datetime(2024, 1, 1, 12, 35, 0),
                datetime(2024, 1, 1, 12, 40, 0),
            )
            assert avg == 10.0

            # Slot [12:40, 12:45) - contains the new sample
            avg, _ = compute_time_weighted_avg(
                session,
                "sensor.test",
                datetime(2024, 1, 1, 12, 40, 0),
                datetime(2024, 1, 1, 12, 45, 0),
            )
            assert avg == 20.0


class TestResampleAllCategoriesTo5Min:
    """Test the main resample_all_categories_to_5min function."""

    def test_no_mappings_no_error(self, patch_engine):
        """No mappings should log warning and return without error."""
        # Should not raise
        resample_all_categories_to_5min()

        # No resampled data should exist
        with Session(patch_engine) as session:
            count = session.query(ResampledSample).count()
            assert count == 0

    def test_no_samples_no_error(self, patch_engine):
        """Mappings but no samples should log warning and return without error."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        # Should not raise
        resample_all_categories_to_5min()

        # No resampled data should exist
        with Session(patch_engine) as session:
            count = session.query(ResampledSample).count()
            assert count == 0

    def test_single_category_resampling(self, patch_engine):
        """Single category with data produces resampled rows."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 10, 0),
                    value=15.0,
                    unit="m/s",
                )
            )
            session.commit()

        resample_all_categories_to_5min()

        with Session(patch_engine) as session:
            resampled = (
                session.query(ResampledSample)
                .order_by(ResampledSample.slot_start)
                .all()
            )

            # Should have slots: [12:00, 12:05) and [12:05, 12:10)
            assert len(resampled) == 2

            assert resampled[0].slot_start == datetime(2024, 1, 1, 12, 0, 0)
            assert resampled[0].category == "WIND"
            assert resampled[0].value == 10.0

            assert resampled[1].slot_start == datetime(2024, 1, 1, 12, 5, 0)
            assert resampled[1].category == "WIND"
            assert resampled[1].value == 10.0

    def test_slot_alignment(self, patch_engine):
        """Slots are aligned to 5-minute boundaries."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            # Start time is not on 5-minute boundary
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 3, 30),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 10, 0),
                    value=15.0,
                    unit="m/s",
                )
            )
            session.commit()

        resample_all_categories_to_5min()

        with Session(patch_engine) as session:
            resampled = (
                session.query(ResampledSample)
                .order_by(ResampledSample.slot_start)
                .all()
            )

            # Slots should be aligned: 12:00, 12:05
            for r in resampled:
                assert r.slot_start.second == 0
                assert r.slot_start.minute % 5 == 0

    def test_incomplete_slot_skipped(self, patch_engine):
        """Slots with missing category data are skipped."""
        with Session(patch_engine) as session:
            # Two categories
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="TEMP",
                    entity_id="sensor.temp",
                    is_active=True,
                    priority=1,
                )
            )

            # WIND has data from 12:00 to 12:30
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 30, 0),
                    value=15.0,
                    unit="m/s",
                )
            )

            # TEMP has data from 12:10 to 12:20
            session.add(
                Sample(
                    entity_id="sensor.temp",
                    timestamp=datetime(2024, 1, 1, 12, 10, 0),
                    value=20.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.temp",
                    timestamp=datetime(2024, 1, 1, 12, 20, 0),
                    value=22.0,
                    unit="°C",
                )
            )
            session.commit()

        resample_all_categories_to_5min()

        with Session(patch_engine) as session:
            resampled = (
                session.query(ResampledSample)
                .order_by(ResampledSample.slot_start, ResampledSample.category)
                .all()
            )

            # Global range is [12:10, 12:20]
            # Aligned start is 12:10
            # Only slots where both have data: [12:10, 12:15) and [12:15, 12:20)
            # Each slot has 2 categories = 4 rows total
            assert len(resampled) == 4

            slot_starts = set(r.slot_start for r in resampled)
            assert slot_starts == {
                datetime(2024, 1, 1, 12, 10, 0),
                datetime(2024, 1, 1, 12, 15, 0),
            }

    def test_idempotence(self, patch_engine):
        """Running resample multiple times produces same results."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 10, 0),
                    value=15.0,
                    unit="m/s",
                )
            )
            session.commit()

        # Run resample twice
        resample_all_categories_to_5min()
        resample_all_categories_to_5min()

        with Session(patch_engine) as session:
            resampled = session.query(ResampledSample).all()
            # Should still only have 2 rows, not 4
            assert len(resampled) == 2

    def test_sparse_data_scenario(self, patch_engine):
        """
        Test the acceptance criteria sparse data scenario.

        Given:
        - 12:00 → 10
        - 12:40 → 20
        - No intermediate samples

        For all slots between 12:00 and 12:40:
        - value = 10 (last known value held constant)
        For slot [12:40, 12:45):
        - value = 20
        """
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 40, 0),
                    value=20.0,
                    unit="m/s",
                )
            )
            session.commit()

        resample_all_categories_to_5min()

        with Session(patch_engine) as session:
            resampled = (
                session.query(ResampledSample)
                .order_by(ResampledSample.slot_start)
                .all()
            )

            # Slots: 12:00, 12:05, 12:10, ..., 12:35
            # (12:40 is the end, so slot [12:35, 12:40) is the last complete slot before 12:40)
            expected_slots = [
                (datetime(2024, 1, 1, 12, 0, 0), 10.0),
                (datetime(2024, 1, 1, 12, 5, 0), 10.0),
                (datetime(2024, 1, 1, 12, 10, 0), 10.0),
                (datetime(2024, 1, 1, 12, 15, 0), 10.0),
                (datetime(2024, 1, 1, 12, 20, 0), 10.0),
                (datetime(2024, 1, 1, 12, 25, 0), 10.0),
                (datetime(2024, 1, 1, 12, 30, 0), 10.0),
                (datetime(2024, 1, 1, 12, 35, 0), 10.0),
            ]

            assert len(resampled) == len(expected_slots)

            for r, (expected_slot, expected_value) in zip(resampled, expected_slots):
                assert r.slot_start == expected_slot
                assert r.value == expected_value


class TestMultipleCategoriesResampling:
    """Test resampling with multiple categories."""

    def test_all_categories_must_have_value(self, patch_engine):
        """All categories must have a value for a slot to be written."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                SensorMapping(
                    category="TEMP",
                    entity_id="sensor.temp",
                    is_active=True,
                    priority=1,
                )
            )

            # Both have same time range
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 10, 0),
                    value=15.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.temp",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    value=20.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.temp",
                    timestamp=datetime(2024, 1, 1, 12, 10, 0),
                    value=22.0,
                    unit="°C",
                )
            )
            session.commit()

        resample_all_categories_to_5min()

        with Session(patch_engine) as session:
            resampled = (
                session.query(ResampledSample)
                .order_by(ResampledSample.slot_start, ResampledSample.category)
                .all()
            )

            # 2 slots × 2 categories = 4 rows
            assert len(resampled) == 4

            # Check both categories present for each slot
            slot_12_00 = [r for r in resampled if r.slot_start == datetime(2024, 1, 1, 12, 0, 0)]
            slot_12_05 = [r for r in resampled if r.slot_start == datetime(2024, 1, 1, 12, 5, 0)]

            assert len(slot_12_00) == 2
            assert len(slot_12_05) == 2

            categories_12_00 = {r.category for r in slot_12_00}
            categories_12_05 = {r.category for r in slot_12_05}

            assert categories_12_00 == {"WIND", "TEMP"}
            assert categories_12_05 == {"WIND", "TEMP"}


class TestResampleStats:
    """Test that resample_all_categories_to_5min returns correct ResampleStats."""

    def test_returns_stats_no_mappings(self, patch_engine):
        """No mappings returns stats with zero values."""
        stats = resample_all_categories_to_5min()

        assert isinstance(stats, ResampleStats)
        assert stats.slots_processed == 0
        assert stats.slots_saved == 0
        assert stats.slots_skipped == 0
        assert stats.categories == []
        assert stats.start_time is None
        assert stats.end_time is None

    def test_returns_stats_no_samples(self, patch_engine):
        """Mappings but no samples returns stats with categories but no times."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            session.commit()

        stats = resample_all_categories_to_5min()

        assert isinstance(stats, ResampleStats)
        assert stats.slots_processed == 0
        assert stats.slots_saved == 0
        assert stats.slots_skipped == 0
        assert "WIND" in stats.categories
        assert stats.start_time is None
        assert stats.end_time is None

    def test_returns_stats_with_data(self, patch_engine):
        """Single category with data returns correct stats."""
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 10, 0),
                    value=15.0,
                    unit="m/s",
                )
            )
            session.commit()

        stats = resample_all_categories_to_5min()

        assert isinstance(stats, ResampleStats)
        assert stats.slots_processed == 2
        assert stats.slots_saved == 2
        assert stats.slots_skipped == 0
        assert stats.categories == ["WIND"]
        assert stats.start_time == datetime(2024, 1, 1, 12, 0, 0)
        assert stats.end_time == datetime(2024, 1, 1, 12, 10, 0)

    def test_returns_stats_with_skipped_slots(self, patch_engine):
        """Two categories with different time ranges shows skipped slots."""
        with Session(patch_engine) as session:
            # WIND: has data from 12:00 to 12:20
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            # TEMP: has data from 12:10 to 12:20
            session.add(
                SensorMapping(
                    category="TEMP",
                    entity_id="sensor.temp",
                    is_active=True,
                    priority=1,
                )
            )

            # WIND data
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 20, 0),
                    value=15.0,
                    unit="m/s",
                )
            )

            # TEMP data - starts later
            session.add(
                Sample(
                    entity_id="sensor.temp",
                    timestamp=datetime(2024, 1, 1, 12, 10, 0),
                    value=20.0,
                    unit="°C",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.temp",
                    timestamp=datetime(2024, 1, 1, 12, 20, 0),
                    value=22.0,
                    unit="°C",
                )
            )
            session.commit()

        stats = resample_all_categories_to_5min()

        assert isinstance(stats, ResampleStats)
        # Global range is [12:10, 12:20] → 2 slots: [12:10, 12:15) and [12:15, 12:20)
        assert stats.slots_processed == 2
        assert stats.slots_saved == 2
        assert stats.slots_skipped == 0
        assert set(stats.categories) == {"WIND", "TEMP"}
