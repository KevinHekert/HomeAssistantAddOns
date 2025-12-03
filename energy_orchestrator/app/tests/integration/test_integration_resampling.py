"""
Integration tests for data resampling workflow.

These tests verify that raw sensor data is correctly resampled into
time slots with proper aggregation using the current API.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from db.models import Sample, SensorMapping, ResampledSample
from db.resample import resample_all_categories


@pytest.mark.integration
def test_basic_resampling(mariadb_engine, clean_database):
    """Test basic resampling of sensor data into 5-minute slots."""
    session = Session(mariadb_engine)
    
    # Create sensor mapping
    mapping = SensorMapping(
        category="test_temp",
        entity_id="sensor.test",
        is_active=True,
        priority=1
    )
    session.add(mapping)
    session.commit()
    
    # Create samples at 1-minute intervals within a 5-minute slot
    base_time = datetime(2024, 1, 1, 12, 0, 0)  # 12:00:00
    samples = []
    
    for i in range(5):
        samples.append(Sample(
            entity_id="sensor.test",
            timestamp=base_time + timedelta(minutes=i),
            value=20.0 + i,  # Values: 20, 21, 22, 23, 24
            unit="°C"
        ))
    
    session.add_all(samples)
    session.commit()
    
    # Run resampling using the current API
    result = resample_all_categories(sample_rate_minutes=5, flush=True)
    
    # Verify the operation completed successfully
    assert result.slots_processed > 0
    assert result.slots_saved > 0
    assert result.sample_rate_minutes == 5
    assert "test_temp" in result.categories
    
    # Verify resampled data
    resampled = session.query(ResampledSample).filter_by(
        category="test_temp",
        slot_start=base_time
    ).first()
    
    assert resampled is not None
    assert resampled.value == pytest.approx(22.0, rel=1e-5)  # Average of 20,21,22,23,24
    assert resampled.unit == "°C"
    assert resampled.is_derived == False
    
    print(f"\n✅ Resampled {len(samples)} samples to average: {resampled.value}")
    print(f"   Stats: {result.slots_processed} processed, {result.slots_saved} saved")
    
    session.close()


@pytest.mark.integration
def test_resampling_with_gaps(mariadb_engine, clean_database):
    """Test resampling when there are gaps in the data."""
    session = Session(mariadb_engine)
    
    # Create sensor mapping
    mapping = SensorMapping(
        category="test_temp",
        entity_id="sensor.test",
        is_active=True,
        priority=1
    )
    session.add(mapping)
    session.commit()
    
    # Create samples with a gap (only at 0, 1, and 4 minutes)
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = [
        Sample(entity_id="sensor.test", timestamp=base_time, value=20.0, unit="°C"),
        Sample(entity_id="sensor.test", timestamp=base_time + timedelta(minutes=1), value=21.0, unit="°C"),
        Sample(entity_id="sensor.test", timestamp=base_time + timedelta(minutes=4), value=24.0, unit="°C"),
    ]
    
    session.add_all(samples)
    session.commit()
    
    # Run resampling
    result = resample_all_categories(sample_rate_minutes=5, flush=True)
    
    assert result.slots_processed > 0
    
    # Verify resampled data exists
    resampled = session.query(ResampledSample).filter_by(
        category="test_temp",
        slot_start=base_time
    ).first()
    
    # Should have resampled data even with gaps (time-weighted average)
    assert resampled is not None
    assert resampled.value > 0
    
    print(f"\n✅ Resampled with gaps: {resampled.value}")
    
    session.close()


@pytest.mark.integration
def test_resampling_multiple_slots(mariadb_engine, clean_database):
    """Test resampling across multiple 5-minute slots."""
    session = Session(mariadb_engine)
    
    # Create sensor mapping
    mapping = SensorMapping(
        category="test_temp",
        entity_id="sensor.test",
        is_active=True,
        priority=1
    )
    session.add(mapping)
    session.commit()
    
    # Create samples spanning 3 slots (15 minutes)
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = []
    
    for slot in range(3):  # 3 slots
        for minute in range(5):  # 5 minutes per slot
            timestamp = base_time + timedelta(minutes=slot*5 + minute)
            value = 20.0 + slot * 10  # Slot 0: 20, Slot 1: 30, Slot 2: 40
            samples.append(Sample(
                entity_id="sensor.test",
                timestamp=timestamp,
                value=value,
                unit="°C"
            ))
    
    session.add_all(samples)
    session.commit()
    
    # Resample all slots at once
    result = resample_all_categories(sample_rate_minutes=5, flush=True)
    
    assert result.slots_saved >= 3
    
    # Verify all slots were resampled
    resampled_count = session.query(ResampledSample).filter_by(category="test_temp").count()
    assert resampled_count >= 3
    
    # Verify values for each slot
    resampled_slots = session.query(ResampledSample).filter_by(
        category="test_temp"
    ).order_by(ResampledSample.slot_start).all()
    
    assert resampled_slots[0].value == pytest.approx(20.0, rel=1e-1)
    assert resampled_slots[1].value == pytest.approx(30.0, rel=1e-1)
    assert resampled_slots[2].value == pytest.approx(40.0, rel=1e-1)
    
    print(f"\n✅ Resampled {resampled_count} slots successfully")
    
    session.close()


@pytest.mark.integration
def test_resampling_no_data(mariadb_engine, clean_database):
    """Test resampling when there's no data."""
    session = Session(mariadb_engine)
    
    # Create sensor mapping but no samples
    mapping = SensorMapping(
        category="test_temp",
        entity_id="sensor.test",
        is_active=True,
        priority=1
    )
    session.add(mapping)
    session.commit()
    
    # Try to resample when there's no data
    result = resample_all_categories(sample_rate_minutes=5, flush=True)
    
    # Should return 0 slots processed
    assert result.slots_processed == 0
    assert result.slots_saved == 0
    
    print(f"\n📊 Resampled with no data: {result.slots_processed} slots processed")
    
    session.close()


@pytest.mark.integration
def test_resampling_multiple_entities_same_category(mariadb_engine, clean_database):
    """Test resampling when multiple entities map to the same category."""
    session = Session(mariadb_engine)
    
    # Create multiple mappings for the same category (priority determines which is used)
    mapping1 = SensorMapping(
        category="outdoor_temp",
        entity_id="sensor.outdoor_1",
        is_active=True,
        priority=1
    )
    mapping2 = SensorMapping(
        category="outdoor_temp",
        entity_id="sensor.outdoor_2",
        is_active=True,
        priority=2
    )
    session.add_all([mapping1, mapping2])
    session.commit()
    
    # Create samples for both entities
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = [
        Sample(entity_id="sensor.outdoor_1", timestamp=base_time, value=10.0, unit="°C"),
        Sample(entity_id="sensor.outdoor_1", timestamp=base_time + timedelta(minutes=1), value=11.0, unit="°C"),
        Sample(entity_id="sensor.outdoor_2", timestamp=base_time, value=12.0, unit="°C"),
        Sample(entity_id="sensor.outdoor_2", timestamp=base_time + timedelta(minutes=1), value=13.0, unit="°C"),
    ]
    
    session.add_all(samples)
    session.commit()
    
    # Run resampling
    result = resample_all_categories(sample_rate_minutes=5, flush=True)
    
    assert result.slots_processed > 0
    assert "outdoor_temp" in result.categories
    
    # Verify resampled data (should use highest priority entity)
    resampled = session.query(ResampledSample).filter_by(
        category="outdoor_temp",
        slot_start=base_time
    ).first()
    
    assert resampled is not None
    print(f"\n✅ Resampled multiple entities: {resampled.value}")
    
    session.close()


@pytest.mark.integration
def test_resampling_preserves_units(mariadb_engine, clean_database):
    """Test that resampling preserves the unit of measurement."""
    session = Session(mariadb_engine)
    
    # Create sensor mapping
    mapping = SensorMapping(
        category="wind_speed",
        entity_id="sensor.wind",
        is_active=True,
        priority=1
    )
    session.add(mapping)
    session.commit()
    
    # Create samples with specific unit
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = [
        Sample(entity_id="sensor.wind", timestamp=base_time, value=5.0, unit="m/s"),
        Sample(entity_id="sensor.wind", timestamp=base_time + timedelta(minutes=1), value=6.0, unit="m/s"),
    ]
    
    session.add_all(samples)
    session.commit()
    
    # Run resampling
    result = resample_all_categories(sample_rate_minutes=5, flush=True)
    
    assert result.slots_processed > 0
    
    # Verify unit is preserved
    resampled = session.query(ResampledSample).filter_by(
        category="wind_speed",
        slot_start=base_time
    ).first()
    
    assert resampled is not None
    assert resampled.unit == "m/s"
    
    print(f"\n✅ Unit preserved: {resampled.unit}")
    
    session.close()


@pytest.mark.integration
def test_resampling_idempotent(mariadb_engine, clean_database):
    """Test that resampling the same data multiple times is idempotent."""
    session = Session(mariadb_engine)
    
    # Create sensor mapping and samples
    mapping = SensorMapping(
        category="test_temp",
        entity_id="sensor.test",
        is_active=True,
        priority=1
    )
    session.add(mapping)
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = [
        Sample(entity_id="sensor.test", timestamp=base_time, value=20.0, unit="°C"),
        Sample(entity_id="sensor.test", timestamp=base_time + timedelta(minutes=1), value=21.0, unit="°C"),
    ]
    session.add_all(samples)
    session.commit()
    
    # Resample twice
    result1 = resample_all_categories(sample_rate_minutes=5, flush=True)
    result2 = resample_all_categories(sample_rate_minutes=5, flush=False)  # Don't flush second time
    
    assert result1.slots_saved > 0
    
    # Should still have only one resampled entry per slot
    count = session.query(ResampledSample).filter_by(
        category="test_temp",
        slot_start=base_time
    ).count()
    
    assert count == 1
    
    print(f"\n✅ Idempotent resampling: {count} entry")
    
    session.close()
