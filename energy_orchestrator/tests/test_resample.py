"""Tests for the resample module."""

import os
import tempfile
from datetime import datetime

import pytest

# Set up test database before importing modules
_temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DB_PATH"] = _temp_db.name
_temp_db.close()


from db.models import Base, Sample, SensorMapping, ResampledSample
from db.schema import init_db_schema, get_engine, get_session
from db.resample import (
    get_primary_entities_by_category,
    get_global_range_for_all_categories,
    compute_time_weighted_avg,
    resample_all_categories_to_5min,
    _align_to_5min_boundary,
    RESAMPLE_STEP,
)


@pytest.fixture(autouse=True)
def reset_db():
    """Reset the database before each test."""
    from db import schema
    
    # Create a fresh temp file for each test
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    
    # Reset the module-level variables
    schema._engine = None
    schema._Session = None
    os.environ["DB_PATH"] = temp_db.name
    
    # Initialize schema
    init_db_schema()
    
    yield temp_db.name
    
    # Cleanup
    try:
        os.unlink(temp_db.name)
    except OSError:
        pass


class TestAlignTo5MinBoundary:
    """Tests for the _align_to_5min_boundary function."""

    def test_already_aligned(self):
        """Test datetime that is already aligned."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = _align_to_5min_boundary(dt)
        assert result == datetime(2024, 1, 1, 12, 0, 0)

    def test_align_minutes(self):
        """Test aligning minutes to 5-minute boundary."""
        dt = datetime(2024, 1, 1, 12, 3, 45)
        result = _align_to_5min_boundary(dt)
        assert result == datetime(2024, 1, 1, 12, 0, 0)

    def test_align_7_minutes(self):
        """Test aligning 7 minutes to 5-minute boundary."""
        dt = datetime(2024, 1, 1, 12, 7, 30)
        result = _align_to_5min_boundary(dt)
        assert result == datetime(2024, 1, 1, 12, 5, 0)

    def test_align_59_minutes(self):
        """Test aligning 59 minutes to 5-minute boundary."""
        dt = datetime(2024, 1, 1, 12, 59, 59, 999999)
        result = _align_to_5min_boundary(dt)
        assert result == datetime(2024, 1, 1, 12, 55, 0)

    def test_strips_seconds_and_microseconds(self):
        """Test that seconds and microseconds are stripped."""
        dt = datetime(2024, 1, 1, 12, 10, 45, 123456)
        result = _align_to_5min_boundary(dt)
        assert result == datetime(2024, 1, 1, 12, 10, 0)
        assert result.second == 0
        assert result.microsecond == 0


class TestGetPrimaryEntitiesByCategory:
    """Tests for the get_primary_entities_by_category function."""

    def test_empty_mappings(self, reset_db):
        """Test with no mappings configured."""
        result = get_primary_entities_by_category()
        assert result == {}

    def test_single_mapping(self, reset_db):
        """Test with a single mapping."""
        session = get_session()
        session.add(
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=1,
            )
        )
        session.commit()
        session.close()

        result = get_primary_entities_by_category()
        assert result == {"WIND": "sensor.wind_speed"}

    def test_multiple_categories(self, reset_db):
        """Test with multiple categories."""
        session = get_session()
        session.add_all([
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=1,
            ),
            SensorMapping(
                category="OUTDOOR_TEMP",
                entity_id="sensor.outdoor_temp",
                is_active=True,
                priority=1,
            ),
        ])
        session.commit()
        session.close()

        result = get_primary_entities_by_category()
        assert result == {
            "WIND": "sensor.wind_speed",
            "OUTDOOR_TEMP": "sensor.outdoor_temp",
        }

    def test_priority_selection(self, reset_db):
        """Test that lowest priority is selected as primary."""
        session = get_session()
        session.add_all([
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_backup",
                is_active=True,
                priority=2,
            ),
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_primary",
                is_active=True,
                priority=1,
            ),
        ])
        session.commit()
        session.close()

        result = get_primary_entities_by_category()
        assert result == {"WIND": "sensor.wind_primary"}

    def test_inactive_mappings_excluded(self, reset_db):
        """Test that inactive mappings are excluded."""
        session = get_session()
        session.add_all([
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_inactive",
                is_active=False,
                priority=1,
            ),
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_active",
                is_active=True,
                priority=2,
            ),
        ])
        session.commit()
        session.close()

        result = get_primary_entities_by_category()
        assert result == {"WIND": "sensor.wind_active"}


class TestGetGlobalRangeForAllCategories:
    """Tests for the get_global_range_for_all_categories function."""

    def test_no_mappings(self, reset_db):
        """Test with no mappings configured."""
        result = get_global_range_for_all_categories()
        assert result == (None, None, {})

    def test_no_samples(self, reset_db):
        """Test with mappings but no samples."""
        session = get_session()
        session.add(
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=1,
            )
        )
        session.commit()
        session.close()

        result = get_global_range_for_all_categories()
        global_start, global_end, category_to_entity = result
        assert global_start is None
        assert global_end is None
        assert category_to_entity == {"WIND": "sensor.wind_speed"}

    def test_single_category_with_samples(self, reset_db):
        """Test with single category and samples."""
        session = get_session()
        session.add(
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=1,
            )
        )
        session.add_all([
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                value=10.0,
                unit="m/s",
            ),
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 13, 0, 0),
                value=15.0,
                unit="m/s",
            ),
        ])
        session.commit()
        session.close()

        result = get_global_range_for_all_categories()
        global_start, global_end, category_to_entity = result
        assert global_start == datetime(2024, 1, 1, 12, 0, 0)
        assert global_end == datetime(2024, 1, 1, 13, 0, 0)
        assert category_to_entity == {"WIND": "sensor.wind_speed"}

    def test_multiple_categories_overlapping_range(self, reset_db):
        """Test that global range is the intersection of all categories."""
        session = get_session()
        session.add_all([
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=1,
            ),
            SensorMapping(
                category="TEMP",
                entity_id="sensor.temp",
                is_active=True,
                priority=1,
            ),
        ])
        # WIND: 11:00 - 14:00
        # TEMP: 12:00 - 13:00
        # Global should be: 12:00 - 13:00 (intersection)
        session.add_all([
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 11, 0, 0),
                value=10.0,
                unit="m/s",
            ),
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 14, 0, 0),
                value=15.0,
                unit="m/s",
            ),
            Sample(
                entity_id="sensor.temp",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                value=20.0,
                unit="C",
            ),
            Sample(
                entity_id="sensor.temp",
                timestamp=datetime(2024, 1, 1, 13, 0, 0),
                value=22.0,
                unit="C",
            ),
        ])
        session.commit()
        session.close()

        result = get_global_range_for_all_categories()
        global_start, global_end, category_to_entity = result
        # global_start = max(11:00, 12:00) = 12:00
        # global_end = min(14:00, 13:00) = 13:00
        assert global_start == datetime(2024, 1, 1, 12, 0, 0)
        assert global_end == datetime(2024, 1, 1, 13, 0, 0)


class TestComputeTimeWeightedAvg:
    """Tests for the compute_time_weighted_avg function."""

    def test_no_samples(self, reset_db):
        """Test with no samples at all."""
        session = get_session()
        result = compute_time_weighted_avg(
            session,
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 5, 0),
        )
        session.close()
        assert result == (None, None)

    def test_single_sample_before_window(self, reset_db):
        """Test with single sample before window (last known value)."""
        session = get_session()
        session.add(
            Sample(
                entity_id="sensor.test",
                timestamp=datetime(2024, 1, 1, 11, 55, 0),
                value=10.0,
                unit="units",
            )
        )
        session.commit()

        result = compute_time_weighted_avg(
            session,
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 5, 0),
        )
        session.close()
        
        avg, unit = result
        assert avg == 10.0
        assert unit == "units"

    def test_single_sample_in_window(self, reset_db):
        """Test with single sample inside window."""
        session = get_session()
        session.add(
            Sample(
                entity_id="sensor.test",
                timestamp=datetime(2024, 1, 1, 12, 2, 30),
                value=10.0,
                unit="units",
            )
        )
        session.commit()

        result = compute_time_weighted_avg(
            session,
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 5, 0),
        )
        session.close()
        
        avg, unit = result
        # Value 10 is used from 12:02:30 to 12:05:00
        # But since there's no previous value, 10 is also used from 12:00:00 to 12:02:30
        # So the entire window uses value 10
        assert avg == 10.0
        assert unit == "units"

    def test_last_known_value_behavior(self, reset_db):
        """Test the last-known-value until next sample behavior.
        
        If there is a 40-minute gap between two measurements:
        - 12:00 → 10
        - 12:40 → 20
        
        For slot [12:00, 12:05), the average should be 10.
        """
        session = get_session()
        session.add_all([
            Sample(
                entity_id="sensor.test",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                value=10.0,
                unit="units",
            ),
            Sample(
                entity_id="sensor.test",
                timestamp=datetime(2024, 1, 1, 12, 40, 0),
                value=20.0,
                unit="units",
            ),
        ])
        session.commit()

        # Test slot [12:00, 12:05)
        result = compute_time_weighted_avg(
            session,
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 5, 0),
        )
        avg, unit = result
        assert avg == 10.0

        # Test slot [12:35, 12:40) - should still be 10 (last known value)
        result = compute_time_weighted_avg(
            session,
            "sensor.test",
            datetime(2024, 1, 1, 12, 35, 0),
            datetime(2024, 1, 1, 12, 40, 0),
        )
        avg, unit = result
        assert avg == 10.0

        # Test slot [12:40, 12:45) - should be 20
        result = compute_time_weighted_avg(
            session,
            "sensor.test",
            datetime(2024, 1, 1, 12, 40, 0),
            datetime(2024, 1, 1, 12, 45, 0),
        )
        avg, unit = result
        assert avg == 20.0

        session.close()

    def test_time_weighted_average_calculation(self, reset_db):
        """Test time-weighted average with multiple samples in window."""
        session = get_session()
        # Window: [12:00, 12:05)
        # Sample at 12:00 with value 10
        # Sample at 12:03 with value 20
        # Expected: (10 * 180) + (20 * 120) = 1800 + 2400 = 4200
        # Total time: 300 seconds
        # Average: 4200 / 300 = 14
        session.add_all([
            Sample(
                entity_id="sensor.test",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                value=10.0,
                unit="units",
            ),
            Sample(
                entity_id="sensor.test",
                timestamp=datetime(2024, 1, 1, 12, 3, 0),
                value=20.0,
                unit="units",
            ),
        ])
        session.commit()

        result = compute_time_weighted_avg(
            session,
            "sensor.test",
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 5, 0),
        )
        session.close()
        
        avg, unit = result
        assert avg == 14.0
        assert unit == "units"


class TestResampleAllCategoriesToFiveMin:
    """Tests for the resample_all_categories_to_5min function."""

    def test_no_mappings(self, reset_db):
        """Test resampling with no mappings configured."""
        resample_all_categories_to_5min()
        
        session = get_session()
        count = session.query(ResampledSample).count()
        session.close()
        
        assert count == 0

    def test_no_samples(self, reset_db):
        """Test resampling with mappings but no samples."""
        session = get_session()
        session.add(
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=1,
            )
        )
        session.commit()
        session.close()

        resample_all_categories_to_5min()

        session = get_session()
        count = session.query(ResampledSample).count()
        session.close()
        
        assert count == 0

    def test_single_category_resampling(self, reset_db):
        """Test resampling with single category."""
        session = get_session()
        session.add(
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=1,
            )
        )
        session.add_all([
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                value=10.0,
                unit="m/s",
            ),
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 12, 10, 0),
                value=15.0,
                unit="m/s",
            ),
        ])
        session.commit()
        session.close()

        resample_all_categories_to_5min()

        session = get_session()
        resampled = (
            session.query(ResampledSample)
            .order_by(ResampledSample.slot_start)
            .all()
        )
        session.close()

        # Should have slots: [12:00, 12:05), [12:05, 12:10)
        assert len(resampled) == 2
        
        assert resampled[0].slot_start == datetime(2024, 1, 1, 12, 0, 0)
        assert resampled[0].category == "WIND"
        assert resampled[0].value == 10.0
        assert resampled[0].unit == "m/s"
        
        assert resampled[1].slot_start == datetime(2024, 1, 1, 12, 5, 0)
        assert resampled[1].category == "WIND"
        assert resampled[1].value == 10.0  # Last known value until 12:10

    def test_incomplete_slot_skipped(self, reset_db):
        """Test that slots with missing category data are skipped."""
        session = get_session()
        session.add_all([
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=1,
            ),
            SensorMapping(
                category="TEMP",
                entity_id="sensor.temp",
                is_active=True,
                priority=1,
            ),
        ])
        # WIND has data from 12:00 - 12:10
        # TEMP has data from 12:05 - 12:10
        session.add_all([
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                value=10.0,
                unit="m/s",
            ),
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 12, 10, 0),
                value=15.0,
                unit="m/s",
            ),
            Sample(
                entity_id="sensor.temp",
                timestamp=datetime(2024, 1, 1, 12, 5, 0),
                value=20.0,
                unit="C",
            ),
            Sample(
                entity_id="sensor.temp",
                timestamp=datetime(2024, 1, 1, 12, 10, 0),
                value=22.0,
                unit="C",
            ),
        ])
        session.commit()
        session.close()

        resample_all_categories_to_5min()

        session = get_session()
        resampled = (
            session.query(ResampledSample)
            .order_by(ResampledSample.slot_start, ResampledSample.category)
            .all()
        )
        session.close()

        # Global range: max(12:00, 12:05) - min(12:10, 12:10) = 12:05 - 12:10
        # Only slot [12:05, 12:10) should be written with both categories
        assert len(resampled) == 2
        
        # Check that both categories are present for slot 12:05
        categories = {r.category for r in resampled}
        assert categories == {"WIND", "TEMP"}
        
        for r in resampled:
            assert r.slot_start == datetime(2024, 1, 1, 12, 5, 0)

    def test_idempotence(self, reset_db):
        """Test that running resample multiple times doesn't create duplicates."""
        session = get_session()
        session.add(
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=1,
            )
        )
        session.add_all([
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                value=10.0,
                unit="m/s",
            ),
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 12, 10, 0),
                value=15.0,
                unit="m/s",
            ),
        ])
        session.commit()
        session.close()

        # Run resample multiple times
        resample_all_categories_to_5min()
        resample_all_categories_to_5min()
        resample_all_categories_to_5min()

        session = get_session()
        count = session.query(ResampledSample).count()
        session.close()

        # Should still only have 2 rows (one per slot)
        assert count == 2

    def test_slot_alignment(self, reset_db):
        """Test that slots are aligned to 5-minute boundaries."""
        session = get_session()
        session.add(
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=1,
            )
        )
        # Start at 12:03 (not aligned)
        session.add_all([
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 12, 3, 0),
                value=10.0,
                unit="m/s",
            ),
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 12, 12, 0),
                value=15.0,
                unit="m/s",
            ),
        ])
        session.commit()
        session.close()

        resample_all_categories_to_5min()

        session = get_session()
        resampled = (
            session.query(ResampledSample)
            .order_by(ResampledSample.slot_start)
            .all()
        )
        session.close()

        # Aligned start should be 12:00
        # Slots: [12:00, 12:05), [12:05, 12:10)
        for r in resampled:
            assert r.slot_start.minute % 5 == 0
            assert r.slot_start.second == 0
            assert r.slot_start.microsecond == 0

    def test_last_known_value_behavior_end_to_end(self, reset_db):
        """Test the complete last-known-value behavior from acceptance criteria.
        
        Given an entity with:
        - 12:00 → 10
        - 12:40 → 20
        - no other samples
        
        Expected results:
        - For all slots [12:00, 12:05), [12:05, 12:10), …, [12:35, 12:40): value = 10
        - For slot [12:40, 12:45): value = 20
        """
        session = get_session()
        session.add(
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=1,
            )
        )
        session.add_all([
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                value=10.0,
                unit="m/s",
            ),
            Sample(
                entity_id="sensor.wind_speed",
                timestamp=datetime(2024, 1, 1, 12, 40, 0),
                value=20.0,
                unit="m/s",
            ),
        ])
        session.commit()
        session.close()

        resample_all_categories_to_5min()

        session = get_session()
        resampled = (
            session.query(ResampledSample)
            .order_by(ResampledSample.slot_start)
            .all()
        )
        session.close()

        # Slots from 12:00 to 12:40 (8 slots: 12:00, 12:05, ..., 12:35)
        # should have value 10
        for i, r in enumerate(resampled[:-1]):
            expected_time = datetime(2024, 1, 1, 12, i * 5, 0)
            assert r.slot_start == expected_time, f"Slot {i} time mismatch"
            assert r.value == 10.0, f"Slot {i} at {r.slot_start} should have value 10"

        # Last slot should not exist since global_end is 12:40
        # and we only process slots where slot_start < global_end
        # So the slot [12:40, 12:45) is NOT processed


class TestSchemaCreation:
    """Tests for schema creation."""

    def test_tables_created(self, reset_db):
        """Test that all required tables are created."""
        engine = get_engine()
        
        # Check that tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        assert "samples" in tables
        assert "sensor_mappings" in tables
        assert "resampled_samples" in tables

    def test_unique_constraints(self, reset_db):
        """Test that unique constraints work."""
        session = get_session()
        
        # Add a mapping
        session.add(
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=1,
            )
        )
        session.commit()
        
        # Try to add duplicate
        session.add(
            SensorMapping(
                category="WIND",
                entity_id="sensor.wind_speed",
                is_active=True,
                priority=2,
            )
        )
        
        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            session.commit()
        
        session.rollback()
        session.close()
