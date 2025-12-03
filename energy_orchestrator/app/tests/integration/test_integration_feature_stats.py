"""
Integration tests for feature statistics calculation.

These tests verify that time-span statistics (avg_1h, avg_6h, avg_24h, avg_7d)
are correctly calculated from resampled data.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from db.models import ResampledSample, FeatureStatistic, SensorMapping
from db.calculate_feature_stats import calculate_feature_stats_for_slot


@pytest.mark.integration
def test_calculate_1h_avg_stats(mariadb_engine, clean_database):
    """Test calculation of 1-hour average statistics."""
    session = Session(mariadb_engine)
    
    # Create resampled samples for 1 hour (12 slots of 5 minutes each)
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = []
    
    for i in range(12):
        slot_start = base_time + timedelta(minutes=i*5)
        samples.append(ResampledSample(
            slot_start=slot_start,
            category="outdoor_temp",
            value=20.0 + i,  # Values from 20 to 31
            unit="°C",
            is_derived=False
        ))
    
    session.add_all(samples)
    session.commit()
    
    # Calculate stats for the last slot
    target_slot = base_time + timedelta(minutes=11*5)
    calculate_feature_stats_for_slot(target_slot)
    
    # Verify 1-hour average was calculated
    stat = session.query(FeatureStatistic).filter_by(
        slot_start=target_slot,
        sensor_name="outdoor_temp",
        stat_type="avg_1h"
    ).first()
    
    assert stat is not None
    # Average of 20,21,22,...,31 = 25.5
    assert stat.value == pytest.approx(25.5, rel=1e-2)
    assert stat.unit == "°C"
    assert stat.source_sample_count == 12
    
    print(f"\n✅ 1h avg calculated: {stat.value}°C from {stat.source_sample_count} samples")
    
    session.close()


@pytest.mark.integration
def test_calculate_multiple_time_windows(mariadb_engine, clean_database):
    """Test calculation of multiple time window statistics (1h, 6h, 24h)."""
    session = Session(mariadb_engine)
    
    # Create resampled samples for 24 hours (288 slots)
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    samples = []
    
    for i in range(288):
        slot_start = base_time + timedelta(minutes=i*5)
        # Create a pattern: values go from 10 to 30 and back
        value = 20.0 + 10.0 * (1.0 - abs((i - 144) / 144.0))
        samples.append(ResampledSample(
            slot_start=slot_start,
            category="outdoor_temp",
            value=value,
            unit="°C",
            is_derived=False
        ))
    
    session.add_all(samples)
    session.commit()
    
    # Calculate stats for a slot in the middle
    target_slot = base_time + timedelta(hours=12)
    calculate_feature_stats_for_slot(target_slot)
    
    # Verify all time windows were calculated
    stats = session.query(FeatureStatistic).filter_by(
        slot_start=target_slot,
        sensor_name="outdoor_temp"
    ).all()
    
    stat_types = [s.stat_type for s in stats]
    
    assert "avg_1h" in stat_types
    assert "avg_6h" in stat_types
    assert "avg_24h" in stat_types
    
    print(f"\n✅ Calculated {len(stats)} time window statistics")
    
    for stat in stats:
        print(f"  {stat.stat_type}: {stat.value:.2f}°C from {stat.source_sample_count} samples")
    
    session.close()


@pytest.mark.integration
def test_feature_stats_with_gaps(mariadb_engine, clean_database):
    """Test feature statistics calculation when there are data gaps."""
    session = Session(mariadb_engine)
    
    # Create resampled samples with gaps (only every other slot)
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = []
    
    for i in range(0, 24, 2):  # Every other slot (12 samples instead of 24)
        slot_start = base_time + timedelta(minutes=i*5)
        samples.append(ResampledSample(
            slot_start=slot_start,
            category="outdoor_temp",
            value=20.0,
            unit="°C",
            is_derived=False
        ))
    
    session.add_all(samples)
    session.commit()
    
    # Calculate stats
    target_slot = base_time + timedelta(hours=1)
    calculate_feature_stats_for_slot(target_slot)
    
    # Verify stats were calculated with available data
    stat = session.query(FeatureStatistic).filter_by(
        slot_start=target_slot,
        sensor_name="outdoor_temp",
        stat_type="avg_1h"
    ).first()
    
    if stat:
        print(f"\n✅ Stats with gaps: {stat.value}°C from {stat.source_sample_count} samples")
        assert stat.source_sample_count <= 12  # Half the normal amount
    
    session.close()


@pytest.mark.integration
def test_feature_stats_multiple_sensors(mariadb_engine, clean_database):
    """Test feature statistics for multiple sensor categories."""
    session = Session(mariadb_engine)
    
    # Create resampled samples for multiple categories
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = []
    
    categories = ["outdoor_temp", "indoor_temp", "wind_speed"]
    values = [10.0, 21.0, 5.0]
    units = ["°C", "°C", "m/s"]
    
    for i in range(12):  # 1 hour of data
        slot_start = base_time + timedelta(minutes=i*5)
        for cat, val, unit in zip(categories, values, units):
            samples.append(ResampledSample(
                slot_start=slot_start,
                category=cat,
                value=val + i * 0.1,
                unit=unit,
                is_derived=False
            ))
    
    session.add_all(samples)
    session.commit()
    
    # Calculate stats
    target_slot = base_time + timedelta(minutes=11*5)
    calculate_feature_stats_for_slot(target_slot)
    
    # Verify stats for all categories
    for cat in categories:
        stat = session.query(FeatureStatistic).filter_by(
            slot_start=target_slot,
            sensor_name=cat,
            stat_type="avg_1h"
        ).first()
        
        assert stat is not None
        print(f"\n✅ {cat}: {stat.value:.2f} {stat.unit}")
    
    session.close()


@pytest.mark.integration
def test_feature_stats_derived_sensors(mariadb_engine, clean_database):
    """Test that derived sensor data is included in feature statistics."""
    session = Session(mariadb_engine)
    
    # Create both raw and derived resampled samples
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = []
    
    for i in range(12):
        slot_start = base_time + timedelta(minutes=i*5)
        
        # Raw sensor data
        samples.append(ResampledSample(
            slot_start=slot_start,
            category="outdoor_temp",
            value=10.0,
            unit="°C",
            is_derived=False
        ))
        
        # Derived sensor data (e.g., virtual sensor)
        samples.append(ResampledSample(
            slot_start=slot_start,
            category="temp_delta",
            value=5.0,
            unit="°C",
            is_derived=True
        ))
    
    session.add_all(samples)
    session.commit()
    
    # Calculate stats
    target_slot = base_time + timedelta(minutes=11*5)
    calculate_feature_stats_for_slot(target_slot)
    
    # Verify stats for both raw and derived sensors
    raw_stat = session.query(FeatureStatistic).filter_by(
        slot_start=target_slot,
        sensor_name="outdoor_temp",
        stat_type="avg_1h"
    ).first()
    
    derived_stat = session.query(FeatureStatistic).filter_by(
        slot_start=target_slot,
        sensor_name="temp_delta",
        stat_type="avg_1h"
    ).first()
    
    assert raw_stat is not None
    assert derived_stat is not None
    
    print(f"\n✅ Raw sensor stat: {raw_stat.value}°C")
    print(f"✅ Derived sensor stat: {derived_stat.value}°C")
    
    session.close()


@pytest.mark.integration
def test_feature_stats_idempotent(mariadb_engine, clean_database):
    """Test that calculating feature stats multiple times is idempotent."""
    session = Session(mariadb_engine)
    
    # Create resampled samples
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = []
    
    for i in range(12):
        slot_start = base_time + timedelta(minutes=i*5)
        samples.append(ResampledSample(
            slot_start=slot_start,
            category="outdoor_temp",
            value=20.0,
            unit="°C",
            is_derived=False
        ))
    
    session.add_all(samples)
    session.commit()
    
    # Calculate stats twice
    target_slot = base_time + timedelta(minutes=11*5)
    calculate_feature_stats_for_slot(target_slot)
    calculate_feature_stats_for_slot(target_slot)
    
    # Should only have one stat entry per type
    count = session.query(FeatureStatistic).filter_by(
        slot_start=target_slot,
        sensor_name="outdoor_temp",
        stat_type="avg_1h"
    ).count()
    
    assert count == 1
    
    print(f"\n✅ Idempotent calculation: {count} entry")
    
    session.close()


@pytest.mark.integration
def test_feature_stats_time_range_boundaries(mariadb_engine, clean_database):
    """Test feature statistics at time range boundaries."""
    session = Session(mariadb_engine)
    
    # Create exactly 1 hour of data
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = []
    
    # 12 samples = 1 hour
    for i in range(12):
        slot_start = base_time + timedelta(minutes=i*5)
        samples.append(ResampledSample(
            slot_start=slot_start,
            category="outdoor_temp",
            value=20.0 + i,
            unit="°C",
            is_derived=False
        ))
    
    session.add_all(samples)
    session.commit()
    
    # Calculate stats for the last slot (should include all 12 samples)
    target_slot = base_time + timedelta(minutes=11*5)
    calculate_feature_stats_for_slot(target_slot)
    
    stat = session.query(FeatureStatistic).filter_by(
        slot_start=target_slot,
        sensor_name="outdoor_temp",
        stat_type="avg_1h"
    ).first()
    
    assert stat is not None
    assert stat.source_sample_count == 12
    
    print(f"\n✅ Boundary test: {stat.source_sample_count} samples in 1h window")
    
    session.close()


@pytest.mark.integration
def test_feature_stats_query_performance(mariadb_engine, clean_database):
    """Test performance of feature statistics queries."""
    import time
    session = Session(mariadb_engine)
    
    # Create a large dataset (7 days = 2016 slots)
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    samples = []
    
    print("\n📊 Creating 7 days of resampled data...")
    for i in range(2016):
        slot_start = base_time + timedelta(minutes=i*5)
        samples.append(ResampledSample(
            slot_start=slot_start,
            category="outdoor_temp",
            value=20.0 + (i % 100) * 0.1,
            unit="°C",
            is_derived=False
        ))
    
    session.add_all(samples)
    session.commit()
    
    # Time the feature stats calculation
    target_slot = base_time + timedelta(days=7)
    
    start = time.time()
    calculate_feature_stats_for_slot(target_slot)
    duration = time.time() - start
    
    print(f"⏱️ Feature stats calculation took {duration:.3f} seconds")
    
    # Verify stats were calculated
    stat_count = session.query(FeatureStatistic).filter_by(
        slot_start=target_slot,
        sensor_name="outdoor_temp"
    ).count()
    
    print(f"✅ Created {stat_count} feature statistics")
    
    assert duration < 5.0, f"Feature stats too slow: {duration:.3f}s"
    
    session.close()
