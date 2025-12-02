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
    resample_all_categories,
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
    """Test the main resample_all_categories function."""

    def test_no_mappings_no_error(self, patch_engine):
        """No mappings should log warning and return without error."""
        # Should not raise
        resample_all_categories()

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
        resample_all_categories()

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

        resample_all_categories()

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

        resample_all_categories()

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

        resample_all_categories()

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
        resample_all_categories()
        resample_all_categories()

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

        resample_all_categories()

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

        resample_all_categories()

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
    """Test that resample_all_categories returns correct ResampleStats."""

    def test_returns_stats_no_mappings(self, patch_engine):
        """No mappings returns stats with zero values."""
        stats = resample_all_categories()

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

        stats = resample_all_categories()

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

        stats = resample_all_categories()

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

        stats = resample_all_categories()

        assert isinstance(stats, ResampleStats)
        # Global range is [12:10, 12:20] → 2 slots: [12:10, 12:15) and [12:15, 12:20)
        assert stats.slots_processed == 2
        assert stats.slots_saved == 2
        assert stats.slots_skipped == 0
        assert set(stats.categories) == {"WIND", "TEMP"}


class TestConfigurableSampleRate:
    """Test configurable sample rate functionality with file-based persistence."""

    def test_get_sample_rate_default_no_config_file(self, monkeypatch, tmp_path):
        """Default sample rate is 5 minutes when config file doesn't exist."""
        import db.resample as resample_module
        
        # Point to non-existent config file
        config_path = tmp_path / "nonexistent" / "resample_config.json"
        monkeypatch.setattr(resample_module, "CONFIG_FILE_PATH", config_path)
        
        assert resample_module.get_sample_rate_minutes() == 5

    def test_set_and_get_sample_rate(self, monkeypatch, tmp_path):
        """Sample rate can be set and retrieved from persistent storage."""
        import db.resample as resample_module
        
        config_path = tmp_path / "resample_config.json"
        monkeypatch.setattr(resample_module, "CONFIG_FILE_PATH", config_path)
        
        # Set sample rate
        assert resample_module.set_sample_rate_minutes(10) is True
        
        # Verify it was saved
        assert resample_module.get_sample_rate_minutes() == 10

    def test_set_sample_rate_invalid_rate(self, monkeypatch, tmp_path):
        """Invalid sample rate returns False and doesn't save."""
        import db.resample as resample_module
        
        config_path = tmp_path / "resample_config.json"
        monkeypatch.setattr(resample_module, "CONFIG_FILE_PATH", config_path)
        
        # Try to set invalid rate
        assert resample_module.set_sample_rate_minutes(7) is False
        
        # Default should still be used
        assert resample_module.get_sample_rate_minutes() == 5

    def test_get_sample_rate_valid_divisors(self, monkeypatch, tmp_path):
        """All valid divisors of 60 are accepted."""
        import db.resample as resample_module
        
        config_path = tmp_path / "resample_config.json"
        monkeypatch.setattr(resample_module, "CONFIG_FILE_PATH", config_path)
        
        for rate in resample_module.VALID_SAMPLE_RATES:
            assert resample_module.set_sample_rate_minutes(rate) is True
            assert resample_module.get_sample_rate_minutes() == rate

    def test_get_sample_rate_corrupt_config_file(self, monkeypatch, tmp_path):
        """Corrupt config file defaults to 5."""
        import db.resample as resample_module
        
        config_path = tmp_path / "resample_config.json"
        monkeypatch.setattr(resample_module, "CONFIG_FILE_PATH", config_path)
        
        # Create corrupt JSON file
        config_path.write_text("not valid json {")
        
        assert resample_module.get_sample_rate_minutes() == 5

    def test_get_sample_rate_config_file_missing_key(self, monkeypatch, tmp_path):
        """Config file without sample_rate_minutes defaults to 5."""
        import db.resample as resample_module
        import json
        
        config_path = tmp_path / "resample_config.json"
        monkeypatch.setattr(resample_module, "CONFIG_FILE_PATH", config_path)
        
        # Create config file without sample_rate_minutes
        config_path.write_text(json.dumps({"other_key": "value"}))
        
        assert resample_module.get_sample_rate_minutes() == 5

    def test_set_sample_rate_creates_parent_directories(self, monkeypatch, tmp_path):
        """set_sample_rate_minutes creates parent directories if they don't exist."""
        import db.resample as resample_module
        
        config_path = tmp_path / "nested" / "dir" / "resample_config.json"
        monkeypatch.setattr(resample_module, "CONFIG_FILE_PATH", config_path)
        
        assert resample_module.set_sample_rate_minutes(15) is True
        assert config_path.exists()
        assert resample_module.get_sample_rate_minutes() == 15

    def test_set_sample_rate_preserves_other_config(self, monkeypatch, tmp_path):
        """set_sample_rate_minutes preserves other keys in config file."""
        import db.resample as resample_module
        import json
        
        config_path = tmp_path / "resample_config.json"
        monkeypatch.setattr(resample_module, "CONFIG_FILE_PATH", config_path)
        
        # Create config file with other settings
        config_path.write_text(json.dumps({"other_setting": "value", "sample_rate_minutes": 5}))
        
        # Update sample rate
        assert resample_module.set_sample_rate_minutes(10) is True
        
        # Verify other settings are preserved
        with open(config_path) as f:
            config = json.load(f)
        assert config["other_setting"] == "value"
        assert config["sample_rate_minutes"] == 10


class TestAlignToBoundary:
    """Test the generic _align_to_boundary function."""

    def test_align_to_5min_boundary(self):
        """Align to 5-minute boundary."""
        from db.resample import _align_to_boundary
        dt = datetime(2024, 1, 1, 12, 7, 30)
        assert _align_to_boundary(dt, 5) == datetime(2024, 1, 1, 12, 5, 0)

    def test_align_to_10min_boundary(self):
        """Align to 10-minute boundary."""
        from db.resample import _align_to_boundary
        dt = datetime(2024, 1, 1, 12, 7, 30)
        assert _align_to_boundary(dt, 10) == datetime(2024, 1, 1, 12, 0, 0)
        
        dt2 = datetime(2024, 1, 1, 12, 15, 45)
        assert _align_to_boundary(dt2, 10) == datetime(2024, 1, 1, 12, 10, 0)

    def test_align_to_15min_boundary(self):
        """Align to 15-minute boundary."""
        from db.resample import _align_to_boundary
        dt = datetime(2024, 1, 1, 12, 20, 30)
        assert _align_to_boundary(dt, 15) == datetime(2024, 1, 1, 12, 15, 0)

    def test_align_to_30min_boundary(self):
        """Align to 30-minute boundary."""
        from db.resample import _align_to_boundary
        dt = datetime(2024, 1, 1, 12, 45, 30)
        assert _align_to_boundary(dt, 30) == datetime(2024, 1, 1, 12, 30, 0)

    def test_align_to_60min_boundary(self):
        """Align to 60-minute (1 hour) boundary."""
        from db.resample import _align_to_boundary
        dt = datetime(2024, 1, 1, 12, 45, 30)
        assert _align_to_boundary(dt, 60) == datetime(2024, 1, 1, 12, 0, 0)

    def test_align_already_aligned(self):
        """Datetime already on boundary stays the same."""
        from db.resample import _align_to_boundary
        dt = datetime(2024, 1, 1, 12, 10, 0)
        assert _align_to_boundary(dt, 10) == datetime(2024, 1, 1, 12, 10, 0)

    def test_align_invalid_rate_defaults_to_5(self):
        """Invalid sample rate (0 or negative) defaults to 5."""
        from db.resample import _align_to_boundary
        dt = datetime(2024, 1, 1, 12, 7, 30)
        assert _align_to_boundary(dt, 0) == datetime(2024, 1, 1, 12, 5, 0)
        assert _align_to_boundary(dt, -5) == datetime(2024, 1, 1, 12, 5, 0)


class TestResampleAllCategoriesWithConfigurableRate:
    """Test resample_all_categories with custom sample rate."""

    def test_resample_with_10min_rate(self, patch_engine):
        """Resample with 10-minute sample rate produces correct slots."""
        from db.resample import resample_all_categories
        
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
                    timestamp=datetime(2024, 1, 1, 12, 20, 0),
                    value=15.0,
                    unit="m/s",
                )
            )
            session.commit()

        stats = resample_all_categories(sample_rate_minutes=10)

        # With 10-min rate: slots [12:00, 12:10) and [12:10, 12:20)
        assert stats.slots_processed == 2
        assert stats.slots_saved == 2
        assert stats.sample_rate_minutes == 10

        with Session(patch_engine) as session:
            resampled = (
                session.query(ResampledSample)
                .order_by(ResampledSample.slot_start)
                .all()
            )
            assert len(resampled) == 2
            assert resampled[0].slot_start == datetime(2024, 1, 1, 12, 0, 0)
            assert resampled[1].slot_start == datetime(2024, 1, 1, 12, 10, 0)

    def test_resample_with_15min_rate(self, patch_engine):
        """Resample with 15-minute sample rate produces correct slots."""
        from db.resample import resample_all_categories
        
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
                    timestamp=datetime(2024, 1, 1, 12, 30, 0),
                    value=15.0,
                    unit="m/s",
                )
            )
            session.commit()

        stats = resample_all_categories(sample_rate_minutes=15)

        # With 15-min rate: slots [12:00, 12:15) and [12:15, 12:30)
        assert stats.slots_processed == 2
        assert stats.slots_saved == 2
        assert stats.sample_rate_minutes == 15

    def test_resample_stats_includes_sample_rate(self, patch_engine):
        """ResampleStats includes sample_rate_minutes field."""
        from db.resample import resample_all_categories
        
        stats = resample_all_categories(sample_rate_minutes=10)
        assert stats.sample_rate_minutes == 10

    def test_resample_uses_config_file_when_no_arg(self, patch_engine, monkeypatch, tmp_path):
        """When no sample_rate_minutes arg, uses persistent config file."""
        import db.resample as resample_module
        from db.resample import resample_all_categories
        
        # Set up config file with 15-minute rate
        config_path = tmp_path / "resample_config.json"
        monkeypatch.setattr(resample_module, "CONFIG_FILE_PATH", config_path)
        resample_module.set_sample_rate_minutes(15)
        
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
                    timestamp=datetime(2024, 1, 1, 12, 30, 0),
                    value=15.0,
                    unit="m/s",
                )
            )
            session.commit()

        stats = resample_all_categories()
        assert stats.sample_rate_minutes == 15

    def test_time_weighted_average_with_longer_window(self, patch_engine):
        """Time-weighted average works correctly with 10-minute windows."""
        from db.resample import resample_all_categories
        
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            # Value 10 at 12:00, value 20 at 12:05
            # For 10-minute window [12:00, 12:10):
            # - 12:00 to 12:05 = 300 seconds at value 10
            # - 12:05 to 12:10 = 300 seconds at value 20
            # Average = (10*300 + 20*300) / 600 = 15
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
                    timestamp=datetime(2024, 1, 1, 12, 5, 0),
                    value=20.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 10, 0),
                    value=30.0,
                    unit="m/s",
                )
            )
            session.commit()

        stats = resample_all_categories(sample_rate_minutes=10)

        with Session(patch_engine) as session:
            resampled = (
                session.query(ResampledSample)
                .filter(ResampledSample.slot_start == datetime(2024, 1, 1, 12, 0, 0))
                .first()
            )
            # Average should be 15 (time-weighted)
            assert resampled is not None
            assert resampled.value == 15.0


class TestFlushResampledSamples:
    """Test the flush_resampled_samples function."""

    def test_flush_empty_table(self, patch_engine):
        """Flushing an empty table returns 0."""
        from db.resample import flush_resampled_samples
        count = flush_resampled_samples()
        assert count == 0

    def test_flush_with_data(self, patch_engine):
        """Flushing a table with data returns the row count and clears the table."""
        from db.resample import flush_resampled_samples
        
        # Add some resampled data
        with Session(patch_engine) as session:
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 12, 0, 0),
                    category="WIND",
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 12, 5, 0),
                    category="WIND",
                    value=15.0,
                    unit="m/s",
                )
            )
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 12, 0, 0),
                    category="TEMP",
                    value=20.0,
                    unit="°C",
                )
            )
            session.commit()

        count = flush_resampled_samples()
        assert count == 3

        # Verify table is empty
        with Session(patch_engine) as session:
            remaining = session.query(ResampledSample).count()
            assert remaining == 0

    def test_flush_before_resample(self, patch_engine):
        """Resample with flush=True clears existing data first."""
        from db.resample import resample_all_categories
        
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            # Add stale resampled data (from old sample rate)
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 11, 0, 0),
                    category="WIND",
                    value=5.0,
                    unit="m/s",
                )
            )
            # Add raw samples
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

        # Resample with flush
        stats = resample_all_categories(sample_rate_minutes=5, flush=True)
        
        assert stats.table_flushed is True
        
        # Verify old data is gone and only new data exists
        with Session(patch_engine) as session:
            resampled = session.query(ResampledSample).all()
            # Should have 2 new slots, not 3 (2 new + 1 old)
            assert len(resampled) == 2
            # Old slot at 11:00 should not exist
            old_slot = session.query(ResampledSample).filter(
                ResampledSample.slot_start == datetime(2024, 1, 1, 11, 0, 0)
            ).first()
            assert old_slot is None

    def test_resample_without_flush(self, patch_engine):
        """Resample with flush=False (default) does not clear existing data."""
        from db.resample import resample_all_categories
        
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            # Add existing resampled data that won't be overwritten
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 11, 0, 0),
                    category="WIND",
                    value=5.0,
                    unit="m/s",
                )
            )
            # Add raw samples
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

        # Resample without flush (default)
        stats = resample_all_categories(sample_rate_minutes=5)
        
        assert stats.table_flushed is False
        
        # Verify old data still exists plus new data
        with Session(patch_engine) as session:
            resampled = session.query(ResampledSample).all()
            # Should have 3 rows: 1 old + 2 new
            assert len(resampled) == 3
            # Old slot at 11:00 should still exist
            old_slot = session.query(ResampledSample).filter(
                ResampledSample.slot_start == datetime(2024, 1, 1, 11, 0, 0)
            ).first()
            assert old_slot is not None
            assert old_slot.value == 5.0

    def test_stats_includes_table_flushed_field(self, patch_engine):
        """ResampleStats includes table_flushed field."""
        from db.resample import resample_all_categories, ResampleStats
        
        stats = resample_all_categories(sample_rate_minutes=5, flush=True)
        
        assert isinstance(stats, ResampleStats)
        assert hasattr(stats, 'table_flushed')
        assert stats.table_flushed is True
        
        stats = resample_all_categories(sample_rate_minutes=5, flush=False)
        assert stats.table_flushed is False


class TestGetLatestResampledSlotStart:
    """Test the get_latest_resampled_slot_start function."""

    def test_empty_table_returns_none(self, patch_engine):
        """Empty resampled_samples table returns None."""
        from db.resample import get_latest_resampled_slot_start
        
        result = get_latest_resampled_slot_start()
        assert result is None

    def test_single_row_returns_correct_slot(self, patch_engine):
        """Single row returns that slot_start."""
        from db.resample import get_latest_resampled_slot_start
        
        with Session(patch_engine) as session:
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 12, 0, 0),
                    category="WIND",
                    value=10.0,
                    unit="m/s",
                )
            )
            session.commit()
        
        result = get_latest_resampled_slot_start()
        assert result == datetime(2024, 1, 1, 12, 0, 0)

    def test_multiple_rows_returns_max_slot(self, patch_engine):
        """Multiple rows returns the maximum slot_start."""
        from db.resample import get_latest_resampled_slot_start
        
        with Session(patch_engine) as session:
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 12, 0, 0),
                    category="WIND",
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 12, 55, 0),
                    category="WIND",
                    value=15.0,
                    unit="m/s",
                )
            )
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 12, 30, 0),
                    category="WIND",
                    value=12.0,
                    unit="m/s",
                )
            )
            session.commit()
        
        result = get_latest_resampled_slot_start()
        assert result == datetime(2024, 1, 1, 12, 55, 0)


class TestIncrementalResampling:
    """Test incremental resampling behavior."""

    def test_incremental_resample_starts_from_correct_position(self, patch_engine):
        """
        When resampling without flush and existing resampled data exists,
        should start from (latest_slot - 2*sample_rate) instead of global start.
        
        Example from issue:
        - Sample rate: 5 minutes
        - Latest resampled slot: 12:55
        - Incremental start should be: 12:55 - (2*5) = 12:45
        """
        from db.resample import resample_all_categories
        
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            # Raw samples from 12:00 to 13:00
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
                    timestamp=datetime(2024, 1, 1, 13, 0, 0),
                    value=20.0,
                    unit="m/s",
                )
            )
            # Existing resampled data up to 12:55
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 12, 55, 0),
                    category="WIND",
                    value=10.0,
                    unit="m/s",
                )
            )
            session.commit()

        # Run resample without flush (incremental mode)
        stats = resample_all_categories(sample_rate_minutes=5, flush=False)

        # Should start from 12:55 - 10 minutes = 12:45
        assert stats.start_time == datetime(2024, 1, 1, 12, 45, 0)
        # Should process 3 slots: [12:45, 12:50), [12:50, 12:55), [12:55, 13:00)
        assert stats.slots_processed == 3
        assert stats.table_flushed is False

    def test_flush_ignores_incremental_start(self, patch_engine):
        """When flush=True, should use global start even if resampled data exists."""
        from db.resample import resample_all_categories
        
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            # Raw samples from 12:00 to 12:30
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
                    value=20.0,
                    unit="m/s",
                )
            )
            # Existing resampled data at 12:25
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 12, 25, 0),
                    category="WIND",
                    value=10.0,
                    unit="m/s",
                )
            )
            session.commit()

        # Run resample with flush
        stats = resample_all_categories(sample_rate_minutes=5, flush=True)

        # Should start from global start (12:00), not incremental start
        assert stats.start_time == datetime(2024, 1, 1, 12, 0, 0)
        assert stats.table_flushed is True

    def test_incremental_uses_global_start_when_no_resampled_data(self, patch_engine):
        """When no resampled data exists, incremental mode uses global start."""
        from db.resample import resample_all_categories
        
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
                    timestamp=datetime(2024, 1, 1, 12, 20, 0),
                    value=20.0,
                    unit="m/s",
                )
            )
            session.commit()

        # Run resample without flush but no existing resampled data
        stats = resample_all_categories(sample_rate_minutes=5, flush=False)

        # Should start from global start (12:00)
        assert stats.start_time == datetime(2024, 1, 1, 12, 0, 0)
        # Should process 4 slots: [12:00, 12:05), [12:05, 12:10), [12:10, 12:15), [12:15, 12:20)
        assert stats.slots_processed == 4

    def test_incremental_updates_existing_slots(self, patch_engine):
        """Incremental resampling updates existing slots with new values."""
        from db.resample import resample_all_categories
        
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            # Raw samples
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
                    value=20.0,
                    unit="m/s",
                )
            )
            # Pre-existing resampled data with old value at 12:15
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 12, 15, 0),
                    category="WIND",
                    value=99.0,  # Wrong value that should be replaced
                    unit="m/s",
                )
            )
            session.commit()

        # Run incremental resample
        stats = resample_all_categories(sample_rate_minutes=5, flush=False)
        
        # Incremental start should be: 12:15 - (2*5) = 12:05
        assert stats.start_time == datetime(2024, 1, 1, 12, 5, 0)

        # Verify the slot at 12:15 was updated
        with Session(patch_engine) as session:
            slot_12_15 = session.query(ResampledSample).filter(
                ResampledSample.slot_start == datetime(2024, 1, 1, 12, 15, 0),
                ResampledSample.category == "WIND",
            ).first()
            
            # Value should now be 10.0 (last known value), not 99.0
            assert slot_12_15 is not None
            assert slot_12_15.value == 10.0

    def test_incremental_with_10min_sample_rate(self, patch_engine):
        """
        Test incremental resampling with 10-minute sample rate.
        
        Latest resampled: 12:50
        Sample rate: 10 minutes
        Incremental start: 12:50 - (2*10) = 12:30
        """
        from db.resample import resample_all_categories
        
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            # Raw samples from 12:00 to 13:00
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
                    timestamp=datetime(2024, 1, 1, 13, 0, 0),
                    value=20.0,
                    unit="m/s",
                )
            )
            # Existing resampled data at 12:50
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 12, 50, 0),
                    category="WIND",
                    value=10.0,
                    unit="m/s",
                )
            )
            session.commit()

        # Run resample with 10-min rate without flush
        stats = resample_all_categories(sample_rate_minutes=10, flush=False)

        # Should start from 12:50 - 20 minutes = 12:30
        assert stats.start_time == datetime(2024, 1, 1, 12, 30, 0)
        # Should process 3 slots: [12:30, 12:40), [12:40, 12:50), [12:50, 13:00)
        assert stats.slots_processed == 3

    def test_incremental_start_not_before_global_start(self, patch_engine):
        """
        Incremental start should not be before global start.
        
        If incremental_start < aligned_start, use aligned_start.
        """
        from db.resample import resample_all_categories
        
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="WIND",
                    entity_id="sensor.wind",
                    is_active=True,
                    priority=1,
                )
            )
            # Raw samples from 12:30 to 13:00
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 12, 30, 0),
                    value=10.0,
                    unit="m/s",
                )
            )
            session.add(
                Sample(
                    entity_id="sensor.wind",
                    timestamp=datetime(2024, 1, 1, 13, 0, 0),
                    value=20.0,
                    unit="m/s",
                )
            )
            # Existing resampled data at 12:35 (very close to global start)
            # Incremental start would be 12:35 - 10 = 12:25, but global start is 12:30
            session.add(
                ResampledSample(
                    slot_start=datetime(2024, 1, 1, 12, 35, 0),
                    category="WIND",
                    value=10.0,
                    unit="m/s",
                )
            )
            session.commit()

        # Run resample without flush
        stats = resample_all_categories(sample_rate_minutes=5, flush=False)

        # Incremental start would be 12:25, but global start is 12:30
        # Should use global start (12:30) since it's later
        assert stats.start_time == datetime(2024, 1, 1, 12, 30, 0)
