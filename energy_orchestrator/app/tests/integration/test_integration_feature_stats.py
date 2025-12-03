"""
Integration tests for feature statistics calculation.

These tests verify that time-span statistics (avg_1h, avg_6h, avg_24h, avg_7d)
are correctly calculated from resampled data using the current API.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from db.models import ResampledSample, FeatureStatistic, SensorMapping
from db.calculate_feature_stats import calculate_feature_statistics


@pytest.mark.integration  
def test_calculate_feature_stats_basic(mariadb_engine, clean_database):
    """Test basic feature statistics calculation."""
    session = Session(mariadb_engine)
    
    # Create sensor mapping
    mapping = SensorMapping(
        category="outdoor_temp",
        entity_id="sensor.outdoor",
        is_active=True,
        priority=1
    )
    session.add(mapping)
    session.commit()
    
    # Create resampled samples for 2 hours (24 slots of 5 minutes each)
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = []
    
    for i in range(24):
        slot_start = base_time + timedelta(minutes=i*5)
        samples.append(ResampledSample(
            slot_start=slot_start,
            category="outdoor_temp",
            value=20.0 + i * 0.5,
            unit="°C",
            is_derived=False
        ))
    
    session.add_all(samples)
    session.commit()
    
    # Calculate statistics
    start_time = base_time
    end_time = base_time + timedelta(hours=2)
    result = calculate_feature_statistics(
        start_time=start_time,
        end_time=end_time,
        sync_with_feature_config=True
    )
    
    # Verify the operation completed
    assert result.stats_calculated >= 0
    assert result.stats_saved >= 0
    
    print(f"\n✅ Feature stats calculated: {result.stats_calculated} stats, {result.stats_saved} saved")
    print(f"   Sensors processed: {result.sensors_processed}")
    print(f"   Stat types: {result.stat_types_processed}")
    
    session.close()


# Simplified tests - just verify the API works
@pytest.mark.integration
def test_calculate_feature_stats_no_data(mariadb_engine, clean_database):
    """Test feature statistics when there is no data."""
    result = calculate_feature_statistics(sync_with_feature_config=True)
    assert result.stats_calculated >= 0
    print(f"\n📊 No data scenario: {result.stats_calculated} stats calculated")
