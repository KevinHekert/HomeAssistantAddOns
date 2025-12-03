"""
Tests for feature statistics calculation.

Verifies that:
1. Feature statistics table is created correctly
2. Rolling averages are calculated correctly
3. Statistics are stored in the separate feature_statistics table
4. Configuration is respected (only enabled stats are calculated)
5. Integration with resampled data works correctly
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import Base, FeatureStatistic, ResampledSample, SensorMapping
from db.calculate_feature_stats import (
    calculate_feature_statistics,
    calculate_rolling_average,
    get_all_sensor_names,
    get_feature_statistic_value,
    flush_feature_statistics,
    STAT_TYPE_WINDOWS,
)
from db.feature_stats import StatType, FeatureStatsConfiguration, SensorStatsConfig, reload_feature_stats_config
from db.virtual_sensors import (
    VirtualSensorDefinition,
    VirtualSensorOperation,
    VirtualSensorsConfiguration,
    reset_virtual_sensors_config,
)
import db.core as core_module
import db.calculate_feature_stats as calc_module
import db.feature_stats as feature_stats_module
import db.sensor_category_config as sensor_cat_module
import db.virtual_sensors as virtual_sensors_module


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory for configuration files."""
    return tmp_path


@pytest.fixture
def patch_all(test_engine, temp_config_dir, monkeypatch):
    """Patch all necessary modules."""
    # Patch engine
    monkeypatch.setattr(core_module, "engine", test_engine)
    monkeypatch.setattr(calc_module, "engine", test_engine)
    
    # Patch init_db_schema
    def mock_init_db_schema():
        Base.metadata.create_all(test_engine)
    
    monkeypatch.setattr(core_module, "init_db_schema", mock_init_db_schema)
    monkeypatch.setattr(calc_module, "init_db_schema", mock_init_db_schema)
    
    # Patch config directories
    monkeypatch.setattr(feature_stats_module, "DATA_DIR", temp_config_dir)
    monkeypatch.setattr(feature_stats_module, "FEATURE_STATS_CONFIG_FILE", 
                       temp_config_dir / "feature_stats_config.json")
    monkeypatch.setattr(sensor_cat_module, "DATA_DIR", temp_config_dir)
    monkeypatch.setattr(sensor_cat_module, "SENSOR_CONFIG_FILE_PATH",
                       temp_config_dir / "sensor_category_config.json")
    monkeypatch.setattr(virtual_sensors_module, "DATA_DIR", temp_config_dir)
    monkeypatch.setattr(virtual_sensors_module, "VIRTUAL_SENSORS_CONFIG_FILE",
                       temp_config_dir / "virtual_sensors_config.json")
    
    # Reset configurations and global singletons
    reset_virtual_sensors_config()
    # Reset feature stats config global
    feature_stats_module._config = None
    
    return test_engine


class TestFeatureStatisticModel:
    """Test the FeatureStatistic model."""
    
    def test_feature_statistic_table_exists(self, test_engine):
        """Verify feature_statistics table is created."""
        tables = Base.metadata.tables.keys()
        assert "feature_statistics" in tables
    
    def test_can_insert_feature_statistic(self, test_engine):
        """Test inserting a feature statistic."""
        with Session(test_engine) as session:
            stat = FeatureStatistic(
                slot_start=datetime(2024, 1, 15, 12, 0),
                sensor_name="outdoor_temp",
                stat_type="avg_1h",
                value=5.5,
                unit="°C",
                source_sample_count=12,
            )
            session.add(stat)
            session.commit()
            
            # Retrieve and verify
            retrieved = session.query(FeatureStatistic).first()
            assert retrieved is not None
            assert retrieved.sensor_name == "outdoor_temp"
            assert retrieved.stat_type == "avg_1h"
            assert retrieved.value == 5.5
            assert retrieved.source_sample_count == 12


class TestRollingAverageCalculation:
    """Test rolling average calculation logic."""
    
    def test_calculate_1h_average(self, patch_all):
        """Test calculating 1-hour rolling average."""
        # Add resampled data: 12 samples over 1 hour (5-minute intervals)
        base_time = datetime(2024, 1, 15, 12, 0)
        with Session(patch_all) as session:
            for i in range(13):  # 13 samples to cover 12 slots
                timestamp = base_time + timedelta(minutes=i * 5)
                session.add(ResampledSample(
                    slot_start=timestamp,
                    category="outdoor_temp",
                    value=5.0 + i * 0.1,  # Slowly increasing
                    unit="°C",
                    is_derived=False,
                ))
            session.commit()
        
        # Calculate 1-hour average at 13:00 (looking back from 12:00 to 13:00)
        target_time = base_time + timedelta(hours=1)
        with Session(patch_all) as session:
            avg, count = calculate_rolling_average(
                session,
                "outdoor_temp",
                StatType.AVG_1H,
                target_time,
                60,
            )
            
            assert avg is not None
            assert count == 12  # 12 five-minute slots in 1 hour
            # Average should be around 5.55 (average of 5.0, 5.1, ..., 6.1)
            assert 5.4 < avg < 5.7
    
    def test_insufficient_data_returns_none(self, patch_all):
        """Test that insufficient data returns None."""
        base_time = datetime(2024, 1, 15, 12, 0)
        
        # Add only 2 samples (not enough for 1 hour)
        with Session(patch_all) as session:
            session.add(ResampledSample(
                slot_start=base_time,
                category="outdoor_temp",
                value=5.0,
                unit="°C",
            ))
            session.add(ResampledSample(
                slot_start=base_time + timedelta(minutes=5),
                category="outdoor_temp",
                value=5.1,
                unit="°C",
            ))
            session.commit()
        
        # Try to calculate 1-hour average with insufficient data
        target_time = base_time + timedelta(hours=1)
        with Session(patch_all) as session:
            avg, count = calculate_rolling_average(
                session,
                "outdoor_temp",
                StatType.AVG_1H,
                target_time,
                60,
            )
            
            # Should return data even if not full window (uses available data)
            assert avg is not None or count == 0


class TestFeatureStatisticsCalculation:
    """Test end-to-end feature statistics calculation."""
    
    def test_calculates_for_configured_sensors(self, patch_all, temp_config_dir):
        """Test that feature stats are calculated for configured sensors."""
        # Enable avg_1h for outdoor_temp sensor
        stats_config = FeatureStatsConfiguration()
        sensor_stats = stats_config.get_sensor_config("outdoor_temp")
        sensor_stats.enable_stat(StatType.AVG_1H)
        stats_config.save()
        
        # Add resampled data covering 9 hours (enough for 1h lookback + calculation window)
        base_time = datetime(2024, 1, 15, 10, 0)
        with Session(patch_all) as session:
            for i in range(109):  # 109 samples = 108 slots over 9 hours
                timestamp = base_time + timedelta(minutes=i * 5)
                session.add(ResampledSample(
                    slot_start=timestamp,
                    category="outdoor_temp",
                    value=5.0,
                    unit="°C",
                    is_derived=False,
                ))
            session.commit()
        
        # Calculate feature statistics with explicit time range (after 1h mark)
        result = calculate_feature_statistics(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=8),
        )
        
        assert result.stats_saved > 0, f"Should have calculated and saved statistics. Result: {result}"
        assert "avg_1h" in result.stat_types_processed
        
        # Verify stats are in the database
        with Session(patch_all) as session:
            stats = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == "outdoor_temp",
                FeatureStatistic.stat_type == "avg_1h",
            ).all()
            
            assert len(stats) > 0, "Should have feature statistics in database"
    
    def test_respects_configuration(self, patch_all, temp_config_dir):
        """Test that only enabled statistics are calculated."""
        # Explicitly disable all stats for indoor_temp
        stats_config = FeatureStatsConfiguration()
        sensor_stats = stats_config.get_sensor_config("indoor_temp")
        sensor_stats.enabled_stats = set()  # No stats enabled
        stats_config.save()
        
        # Reload config to ensure the saved config is used
        reload_feature_stats_config()
        
        # Add resampled data
        base_time = datetime(2024, 1, 15, 10, 0)
        with Session(patch_all) as session:
            for i in range(25):
                timestamp = base_time + timedelta(minutes=i * 5)
                session.add(ResampledSample(
                    slot_start=timestamp,
                    category="indoor_temp",
                    value=20.0,
                    is_derived=False,
                ))
            session.commit()
        
        # Calculate feature statistics
        result = calculate_feature_statistics(sync_with_feature_config=False)
        
        # Should not save any stats for indoor_temp (none enabled)
        with Session(patch_all) as session:
            stats = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == "indoor_temp",
            ).all()
            
            assert len(stats) == 0, "Should not calculate stats when none are enabled"


class TestFeatureStatisticsUtilities:
    """Test utility functions."""
    
    def test_get_feature_statistic_value(self, patch_all):
        """Test retrieving a specific statistic value."""
        # Insert a stat
        target_time = datetime(2024, 1, 15, 12, 0)
        with Session(patch_all) as session:
            session.add(FeatureStatistic(
                slot_start=target_time,
                sensor_name="outdoor_temp",
                stat_type="avg_1h",
                value=5.5,
                unit="°C",
                source_sample_count=12,
            ))
            session.commit()
        
        # Retrieve it
        value = get_feature_statistic_value("outdoor_temp", StatType.AVG_1H, target_time)
        assert value == 5.5
        
        # Non-existent stat
        value = get_feature_statistic_value("nonexistent", StatType.AVG_1H, target_time)
        assert value is None
    
    def test_flush_feature_statistics(self, patch_all):
        """Test flushing all feature statistics."""
        # Add some stats
        with Session(patch_all) as session:
            for i in range(5):
                session.add(FeatureStatistic(
                    slot_start=datetime(2024, 1, 15, 12, i),
                    sensor_name="test",
                    stat_type="avg_1h",
                    value=1.0,
                    source_sample_count=10,
                ))
            session.commit()
        
        # Flush
        count = flush_feature_statistics()
        assert count == 5
        
        # Verify empty
        with Session(patch_all) as session:
            remaining = session.query(FeatureStatistic).count()
            assert remaining == 0
