"""
Test for per-stat-type time range calculation.

This test verifies the new requirement: each stat type (avg_1h, avg_6h, avg_24h, avg_7d)
should calculate independently based on its own window requirements, rather than being
blocked by the largest window size.

For example:
- With 2 hours of data: avg_1h should work
- With 1 day of data: avg_1h, avg_6h, and avg_24h should work  
- With 8 days of data: all stat types including avg_7d should work
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import Base, FeatureStatistic, ResampledSample
from db.calculate_feature_stats import calculate_feature_statistics
from db.feature_stats import StatType, FeatureStatsConfiguration
from db.feature_stats import reset_feature_stats_config
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
    
    # Reset config global by clearing the cache (will reload on next access)
    reset_feature_stats_config()
    
    return test_engine


class TestPerStatTypeTimeRanges:
    """Test that each stat type calculates independently based on its own window."""
    
    def test_2_hours_data_enables_avg_1h_only(self, patch_all, temp_config_dir):
        """
        With 2 hours of data, only avg_1h should produce statistics.
        
        This demonstrates that avg_1h can work even when longer windows can't.
        """
        # Enable all stat types
        stats_config = FeatureStatsConfiguration()
        sensor_stats = stats_config.get_sensor_config("outdoor_temp")
        sensor_stats.enabled_stats = {
            StatType.AVG_1H,
            StatType.AVG_6H,
            StatType.AVG_24H,
            StatType.AVG_7D,
        }
        stats_config.save()
        
        # Add exactly 2 hours of data
        base_time = datetime(2025, 12, 3, 0, 0)
        end_time = base_time + timedelta(hours=2)
        
        with Session(patch_all) as session:
            current_time = base_time
            while current_time <= end_time:
                session.add(ResampledSample(
                    slot_start=current_time,
                    category="outdoor_temp",
                    value=5.0,
                    unit="°C",
                    is_derived=False,
                ))
                current_time += timedelta(minutes=5)
            session.commit()
        
        # Calculate feature statistics
        result = calculate_feature_statistics(sync_with_feature_config=False)
        
        # Verify that calculation completed
        assert result.stats_saved > 0, "Should calculate some statistics"
        
        # Check which stat types were actually calculated
        with Session(patch_all) as session:
            stat_types = session.query(FeatureStatistic.stat_type).filter(
                FeatureStatistic.sensor_name == "outdoor_temp"
            ).distinct().all()
            
            stat_type_values = [st[0] for st in stat_types]
            
            # avg_1h should definitely work with 2 hours of data
            assert "avg_1h" in stat_type_values, "avg_1h should work with 2 hours of data"
            
            # Longer windows should NOT work (not enough history)
            assert "avg_6h" not in stat_type_values, "avg_6h should NOT work with only 2 hours"
            assert "avg_24h" not in stat_type_values, "avg_24h should NOT work with only 2 hours"
            assert "avg_7d" not in stat_type_values, "avg_7d should NOT work with only 2 hours"
    
    def test_2_days_data_enables_short_windows(self, patch_all, temp_config_dir):
        """
        With 2 days of data, avg_1h, avg_6h, and avg_24h should all work.
        Only avg_7d should be blocked.
        """
        # Enable all stat types
        stats_config = FeatureStatsConfiguration()
        sensor_stats = stats_config.get_sensor_config("outdoor_temp")
        sensor_stats.enabled_stats = {
            StatType.AVG_1H,
            StatType.AVG_6H,
            StatType.AVG_24H,
            StatType.AVG_7D,
        }
        stats_config.save()
        
        # Add 2 days of data
        base_time = datetime(2025, 12, 1, 0, 0)
        end_time = base_time + timedelta(days=2)
        
        with Session(patch_all) as session:
            current_time = base_time
            while current_time <= end_time:
                session.add(ResampledSample(
                    slot_start=current_time,
                    category="outdoor_temp",
                    value=5.0,
                    unit="°C",
                    is_derived=False,
                ))
                current_time += timedelta(minutes=5)
            session.commit()
        
        # Calculate feature statistics
        result = calculate_feature_statistics(sync_with_feature_config=False)
        
        # Verify that calculation completed
        assert result.stats_saved > 0, "Should calculate statistics"
        
        # Check which stat types were calculated
        with Session(patch_all) as session:
            stat_types = session.query(FeatureStatistic.stat_type).filter(
                FeatureStatistic.sensor_name == "outdoor_temp"
            ).distinct().all()
            
            stat_type_values = [st[0] for st in stat_types]
            
            # Short windows should all work with 2 days of data
            assert "avg_1h" in stat_type_values, "avg_1h should work with 2 days"
            assert "avg_6h" in stat_type_values, "avg_6h should work with 2 days"
            assert "avg_24h" in stat_type_values, "avg_24h should work with 2 days"
            
            # 7d window should NOT work (not enough history)
            assert "avg_7d" not in stat_type_values, "avg_7d should NOT work with only 2 days"
            
            # Verify we get different numbers of stats for each type
            # (more stats for shorter windows since they can start earlier)
            count_1h = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == "outdoor_temp",
                FeatureStatistic.stat_type == "avg_1h"
            ).count()
            
            count_24h = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == "outdoor_temp",
                FeatureStatistic.stat_type == "avg_24h"
            ).count()
            
            # avg_1h should have more records than avg_24h
            # (can start after 1 hour vs 24 hours)
            assert count_1h > count_24h, (
                f"avg_1h should have more records than avg_24h. "
                f"Got avg_1h={count_1h}, avg_24h={count_24h}"
            )
    
    def test_8_days_data_enables_all_windows(self, patch_all, temp_config_dir):
        """
        With 8 days of data, all stat types including avg_7d should work.
        """
        # Enable all stat types
        stats_config = FeatureStatsConfiguration()
        sensor_stats = stats_config.get_sensor_config("outdoor_temp")
        sensor_stats.enabled_stats = {
            StatType.AVG_1H,
            StatType.AVG_6H,
            StatType.AVG_24H,
            StatType.AVG_7D,
        }
        stats_config.save()
        
        # Add 8 days of data (enough for 7d window)
        base_time = datetime(2025, 11, 25, 0, 0)
        end_time = base_time + timedelta(days=8)
        
        with Session(patch_all) as session:
            current_time = base_time
            # Add data every 15 minutes to reduce test time
            while current_time <= end_time:
                session.add(ResampledSample(
                    slot_start=current_time,
                    category="outdoor_temp",
                    value=5.0,
                    unit="°C",
                    is_derived=False,
                ))
                current_time += timedelta(minutes=15)
            session.commit()
        
        # Calculate feature statistics
        result = calculate_feature_statistics(sync_with_feature_config=False)
        
        # Verify that calculation completed
        assert result.stats_saved > 0, "Should calculate statistics"
        
        # Check which stat types were calculated
        with Session(patch_all) as session:
            stat_types = session.query(FeatureStatistic.stat_type).filter(
                FeatureStatistic.sensor_name == "outdoor_temp"
            ).distinct().all()
            
            stat_type_values = [st[0] for st in stat_types]
            
            # All stat types should work with 8 days of data
            assert "avg_1h" in stat_type_values, "avg_1h should work with 8 days"
            assert "avg_6h" in stat_type_values, "avg_6h should work with 8 days"
            assert "avg_24h" in stat_type_values, "avg_24h should work with 8 days"
            assert "avg_7d" in stat_type_values, "avg_7d should work with 8 days"
            
            # Verify each stat type has records
            for stat_type in ["avg_1h", "avg_6h", "avg_24h", "avg_7d"]:
                count = session.query(FeatureStatistic).filter(
                    FeatureStatistic.sensor_name == "outdoor_temp",
                    FeatureStatistic.stat_type == stat_type
                ).count()
                assert count > 0, f"{stat_type} should have records with 8 days of data"
    
    def test_different_stat_counts_per_window(self, patch_all, temp_config_dir):
        """
        Verify that shorter windows produce more statistics than longer windows.
        
        This is because shorter windows can start earlier in the data range.
        """
        # Enable multiple stat types
        stats_config = FeatureStatsConfiguration()
        sensor_stats = stats_config.get_sensor_config("outdoor_temp")
        sensor_stats.enabled_stats = {
            StatType.AVG_1H,
            StatType.AVG_6H,
            StatType.AVG_24H,
        }
        stats_config.save()
        
        # Add 3 days of data
        base_time = datetime(2025, 12, 1, 0, 0)
        end_time = base_time + timedelta(days=3)
        
        with Session(patch_all) as session:
            current_time = base_time
            while current_time <= end_time:
                session.add(ResampledSample(
                    slot_start=current_time,
                    category="outdoor_temp",
                    value=5.0,
                    unit="°C",
                    is_derived=False,
                ))
                current_time += timedelta(minutes=5)
            session.commit()
        
        # Calculate feature statistics
        result = calculate_feature_statistics(sync_with_feature_config=False)
        
        # Get counts for each stat type
        with Session(patch_all) as session:
            count_1h = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == "outdoor_temp",
                FeatureStatistic.stat_type == "avg_1h"
            ).count()
            
            count_6h = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == "outdoor_temp",
                FeatureStatistic.stat_type == "avg_6h"
            ).count()
            
            count_24h = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == "outdoor_temp",
                FeatureStatistic.stat_type == "avg_24h"
            ).count()
            
            # Verify the pattern: shorter windows have more records
            assert count_1h > count_6h, (
                f"avg_1h should have more records than avg_6h. "
                f"Got 1h={count_1h}, 6h={count_6h}"
            )
            assert count_6h > count_24h, (
                f"avg_6h should have more records than avg_24h. "
                f"Got 6h={count_6h}, 24h={count_24h}"
            )
            
            # All should have at least some records
            assert count_1h > 0, "avg_1h should have records"
            assert count_6h > 0, "avg_6h should have records"
            assert count_24h > 0, "avg_24h should have records"
