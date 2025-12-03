"""
Test that feature statistics are calculated when enabled.

This test verifies the fix for the issue where feature_statistics table
stays empty even when derived features like wind_avg_1h are enabled.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import Base, FeatureStatistic, ResampledSample
from db.calculate_feature_stats import calculate_feature_statistics
from db.feature_stats import StatType, FeatureStatsConfiguration, reload_feature_stats_config
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
    
    return test_engine


class TestFeatureStatisticsNotEmpty:
    """Test that feature_statistics table is populated when features are enabled."""
    
    def test_wind_avg_1h_is_calculated(self, patch_all, temp_config_dir):
        """
        Test that wind_avg_1h feature statistics are calculated and stored.
        
        This reproduces the reported issue where enabling wind_avg_1h
        results in an empty feature_statistics table.
        """
        # Enable ONLY avg_1h for wind sensor (clear defaults first)
        stats_config = FeatureStatsConfiguration()
        wind_config = stats_config.get_sensor_config("wind")
        wind_config.enabled_stats = {StatType.AVG_1H}  # Clear defaults, set only AVG_1H
        stats_config.save()
        
        # Add 2 hours of wind data in resampled_samples
        base_time = datetime(2024, 1, 15, 10, 0)
        with Session(patch_all) as session:
            for i in range(25):  # 25 samples = 24 slots over 2 hours
                timestamp = base_time + timedelta(minutes=i * 5)
                session.add(ResampledSample(
                    slot_start=timestamp,
                    category="wind",
                    value=5.0,
                    unit="m/s",
                    is_derived=False,
                ))
            session.commit()
        
        # Calculate feature statistics (without explicit time range)
        result = calculate_feature_statistics(sync_with_feature_config=False)
        
        print(f"Result: stats_calculated={result.stats_calculated}, stats_saved={result.stats_saved}")
        print(f"Sensors processed: {result.sensors_processed}")
        print(f"Stat types: {result.stat_types_processed}")
        print(f"Time range: {result.start_time} to {result.end_time}")
        
        # Verify that statistics were saved
        assert result.stats_saved > 0, "Feature statistics should be saved"
        assert "avg_1h" in result.stat_types_processed
        
        # Verify feature_statistics table is NOT empty
        with Session(patch_all) as session:
            count = session.query(FeatureStatistic).count()
            print(f"Total feature statistics in DB: {count}")
            assert count > 0, "feature_statistics table should not be empty"
            
            # Verify wind_avg_1h entries exist
            wind_stats = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == "wind",
                FeatureStatistic.stat_type == "avg_1h",
            ).all()
            
            assert len(wind_stats) > 0, "Should have wind_avg_1h entries"
            print(f"Found {len(wind_stats)} wind_avg_1h statistics")
            
            # Verify the entries have reasonable values
            for stat in wind_stats:
                print(f"  {stat.slot_start}: value={stat.value}, count={stat.source_sample_count}")
                assert stat.value is not None
                assert stat.source_sample_count > 0
    
    def test_wind_avg_6h_is_calculated(self, patch_all, temp_config_dir):
        """
        Test that wind_avg_6h feature statistics are calculated and stored.
        """
        # Enable ONLY avg_6h for wind sensor
        stats_config = FeatureStatsConfiguration()
        wind_config = stats_config.get_sensor_config("wind")
        wind_config.enabled_stats = {StatType.AVG_6H}  # Clear defaults, set only AVG_6H
        stats_config.save()
        
        # Reload to ensure it's loaded from disk
        reload_feature_stats_config()
        
        # Add 8 hours of wind data in resampled_samples
        base_time = datetime(2024, 1, 15, 10, 0)
        with Session(patch_all) as session:
            for i in range(97):  # 97 samples = 96 slots over 8 hours
                timestamp = base_time + timedelta(minutes=i * 5)
                session.add(ResampledSample(
                    slot_start=timestamp,
                    category="wind",
                    value=5.0,
                    unit="m/s",
                    is_derived=False,
                ))
            session.commit()
        
        # Calculate feature statistics
        result = calculate_feature_statistics(sync_with_feature_config=False)
        
        print(f"Result: stats_calculated={result.stats_calculated}, stats_saved={result.stats_saved}")
        print(f"Time range: {result.start_time} to {result.end_time}")
        
        # Verify that statistics were saved
        assert result.stats_saved > 0, "Feature statistics should be saved"
        assert "avg_6h" in result.stat_types_processed
        
        # Verify feature_statistics table is NOT empty
        with Session(patch_all) as session:
            wind_stats = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == "wind",
                FeatureStatistic.stat_type == "avg_6h",
            ).all()
            
            assert len(wind_stats) > 0, "Should have wind_avg_6h entries"
            print(f"Found {len(wind_stats)} wind_avg_6h statistics")
    
    def test_multiple_sensors_and_windows(self, patch_all, temp_config_dir):
        """
        Test that multiple sensors with multiple time windows are all calculated.
        """
        # Enable multiple stats for multiple sensors
        stats_config = FeatureStatsConfiguration()
        wind_config = stats_config.get_sensor_config("wind")
        wind_config.enabled_stats = {StatType.AVG_1H, StatType.AVG_6H}  # Clear defaults
        temp_config = stats_config.get_sensor_config("outdoor_temp")
        temp_config.enabled_stats = {StatType.AVG_1H}  # Clear defaults
        stats_config.save()
        
        # Reload to ensure it's loaded from disk
        reload_feature_stats_config()
        
        # Add 8 hours of data for both sensors
        base_time = datetime(2024, 1, 15, 10, 0)
        with Session(patch_all) as session:
            for i in range(97):
                timestamp = base_time + timedelta(minutes=i * 5)
                session.add(ResampledSample(
                    slot_start=timestamp,
                    category="wind",
                    value=5.0,
                    unit="m/s",
                    is_derived=False,
                ))
                session.add(ResampledSample(
                    slot_start=timestamp,
                    category="outdoor_temp",
                    value=10.0,
                    unit="°C",
                    is_derived=False,
                ))
            session.commit()
        
        # Calculate feature statistics
        result = calculate_feature_statistics(sync_with_feature_config=False)
        
        print(f"Sensors processed: {result.sensors_processed}")
        print(f"Stats saved: {result.stats_saved}")
        print(f"Stat types: {result.stat_types_processed}")
        
        # Verify all expected statistics exist
        with Session(patch_all) as session:
            wind_1h = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == "wind",
                FeatureStatistic.stat_type == "avg_1h",
            ).count()
            
            wind_6h = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == "wind",
                FeatureStatistic.stat_type == "avg_6h",
            ).count()
            
            temp_1h = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == "outdoor_temp",
                FeatureStatistic.stat_type == "avg_1h",
            ).count()
            
            print(f"wind_avg_1h: {wind_1h}")
            print(f"wind_avg_6h: {wind_6h}")
            print(f"outdoor_temp_avg_1h: {temp_1h}")
            
            assert wind_1h > 0, "Should have wind_avg_1h statistics"
            assert wind_6h > 0, "Should have wind_avg_6h statistics"
            assert temp_1h > 0, "Should have outdoor_temp_avg_1h statistics"
