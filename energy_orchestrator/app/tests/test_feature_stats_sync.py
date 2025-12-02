"""
Test synchronization between feature statistics and ML feature configuration.

This test verifies that feature statistics (avg_1h, avg_6h, etc.) are only
calculated for sensors when the corresponding ML features are enabled.
"""

import os
import tempfile
import unittest
from pathlib import Path

from db.feature_stats import (
    FeatureStatsConfiguration,
    StatType,
    derive_stats_from_feature_config,
    sync_stats_config_with_features,
)
from ml.feature_config import (
    FeatureConfiguration,
    CORE_FEATURES,
    EXPERIMENTAL_FEATURES,
)


class TestFeatureStatsSync(unittest.TestCase):
    """Test feature statistics sync with ML feature configuration."""
    
    def setUp(self):
        """Set up test environment with temporary data directory."""
        self.test_dir = tempfile.mkdtemp()
        os.environ["DATA_DIR"] = self.test_dir
        os.environ["CONFIG_DIR"] = self.test_dir
        
        # Clear global configs
        import db.feature_stats
        import ml.feature_config
        db.feature_stats._config = None
        ml.feature_config._config = None
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_derive_stats_from_core_features(self):
        """Test deriving required statistics from CORE features."""
        # Default configuration has only CORE features enabled
        required_stats = derive_stats_from_feature_config()
        
        # Verify outdoor_temp needs avg_1h and avg_24h (from CORE features)
        self.assertIn("outdoor_temp", required_stats)
        self.assertIn(StatType.AVG_1H, required_stats["outdoor_temp"])
        self.assertIn(StatType.AVG_24H, required_stats["outdoor_temp"])
        
        # outdoor_temp_avg_6h is EXPERIMENTAL, so it shouldn't be in required stats by default
        self.assertNotIn(StatType.AVG_6H, required_stats["outdoor_temp"])
        
        # Verify indoor_temp needs avg_6h and avg_24h (from CORE features)
        self.assertIn("indoor_temp", required_stats)
        self.assertIn(StatType.AVG_6H, required_stats["indoor_temp"])
        self.assertIn(StatType.AVG_24H, required_stats["indoor_temp"])
        
        # Verify target_temp needs avg_6h (from CORE features)
        self.assertIn("target_temp", required_stats)
        self.assertIn(StatType.AVG_6H, required_stats["target_temp"])
        # target_temp_avg_24h is EXPERIMENTAL, so it shouldn't be required by default
        self.assertNotIn(StatType.AVG_24H, required_stats["target_temp"])
    
    def test_derive_stats_with_experimental_features(self):
        """Test deriving required statistics when EXPERIMENTAL features are enabled."""
        from ml.feature_config import get_feature_config
        
        config = get_feature_config()
        
        # Enable some experimental features
        config.enable_experimental_feature("outdoor_temp_avg_6h")
        config.enable_experimental_feature("target_temp_avg_24h")
        config.save()
        
        required_stats = derive_stats_from_feature_config()
        
        # Now outdoor_temp should need avg_6h
        self.assertIn("outdoor_temp", required_stats)
        self.assertIn(StatType.AVG_6H, required_stats["outdoor_temp"])
        
        # And target_temp should need avg_24h
        self.assertIn("target_temp", required_stats)
        self.assertIn(StatType.AVG_24H, required_stats["target_temp"])
    
    def test_derive_stats_ignores_non_aggregation_features(self):
        """Test that non-aggregation features don't create stat requirements."""
        required_stats = derive_stats_from_feature_config()
        
        # Raw sensor features (outdoor_temp, wind, humidity) shouldn't create stats for themselves
        # They're in the dict because their aggregations exist, but check that raw features work
        
        # Features like heating_kwh_last_1h are calculated in heating_features.py,
        # not as sensor statistics, so they shouldn't appear here
        # (These are derived from hp_kwh_total, not direct aggregations)
        
        # Time features like hour_of_day, day_of_week don't need sensor stats
        self.assertNotIn("hour_of_day", required_stats)
        self.assertNotIn("day_of_week", required_stats)
        self.assertNotIn("is_weekend", required_stats)
    
    def test_sync_stats_config_with_features(self):
        """Test synchronizing feature stats config with ML feature config."""
        from db.feature_stats import get_feature_stats_config
        
        # Start with default stats config (has defaults for all sensors)
        stats_config = get_feature_stats_config()
        
        # Set up some initial stats
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_1H, True)
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_6H, True)
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_24H, True)
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_7D, True)
        stats_config.save()
        
        # Sync with feature config (which by default doesn't have outdoor_temp_avg_7d)
        synced_config = sync_stats_config_with_features()
        
        # After sync, only stats needed by active ML features should be enabled
        outdoor_stats = synced_config.get_sensor_config("outdoor_temp").enabled_stats
        
        # outdoor_temp_avg_1h and outdoor_temp_avg_24h are CORE, so they should be enabled
        self.assertIn(StatType.AVG_1H, outdoor_stats)
        self.assertIn(StatType.AVG_24H, outdoor_stats)
        
        # outdoor_temp_avg_6h is EXPERIMENTAL and not enabled by default
        self.assertNotIn(StatType.AVG_6H, outdoor_stats)
        
        # outdoor_temp_avg_7d is EXPERIMENTAL and not enabled by default
        self.assertNotIn(StatType.AVG_7D, outdoor_stats)
    
    def test_sync_enables_stats_when_experimental_enabled(self):
        """Test that sync enables stats when experimental features are enabled."""
        from ml.feature_config import get_feature_config
        from db.feature_stats import get_feature_stats_config
        
        # Enable experimental feature
        ml_config = get_feature_config()
        ml_config.enable_experimental_feature("outdoor_temp_avg_7d")
        ml_config.save()
        
        # Sync stats config
        sync_stats_config_with_features()
        
        # Verify that avg_7d is now enabled for outdoor_temp
        stats_config = get_feature_stats_config()
        outdoor_stats = stats_config.get_sensor_config("outdoor_temp").enabled_stats
        
        self.assertIn(StatType.AVG_7D, outdoor_stats)
    
    def test_feature_names_with_multiple_underscores(self):
        """Test handling of feature names with multiple underscores."""
        # Some features have names like "delta_target_indoor" which don't follow
        # the sensor_avg_window pattern and shouldn't create stat requirements
        
        required_stats = derive_stats_from_feature_config()
        
        # delta_target_indoor is a derived feature, not a sensor aggregation
        self.assertNotIn("delta", required_stats)
        self.assertNotIn("delta_target", required_stats)


class TestFeatureStatsIntegration(unittest.TestCase):
    """Integration tests for feature statistics."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        os.environ["DATA_DIR"] = self.test_dir
        os.environ["CONFIG_DIR"] = self.test_dir
        
        # Clear global configs
        import db.feature_stats
        import ml.feature_config
        db.feature_stats._config = None
        ml.feature_config._config = None
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_end_to_end_stats_config_respects_ml_features(self):
        """Test end-to-end that stats config respects ML feature configuration."""
        from ml.feature_config import get_feature_config
        from db.feature_stats import get_feature_stats_config
        
        # Start with default ML config (only CORE features)
        ml_config = get_feature_config()
        
        # Sync stats config
        sync_stats_config_with_features()
        
        # Get stats config
        stats_config = get_feature_stats_config()
        
        # Verify that CORE feature aggregations are enabled
        outdoor_stats = stats_config.get_sensor_config("outdoor_temp").enabled_stats
        self.assertIn(StatType.AVG_1H, outdoor_stats)  # outdoor_temp_avg_1h is CORE
        self.assertIn(StatType.AVG_24H, outdoor_stats)  # outdoor_temp_avg_24h is CORE
        
        # Verify that EXPERIMENTAL feature aggregations are NOT enabled by default
        self.assertNotIn(StatType.AVG_6H, outdoor_stats)  # outdoor_temp_avg_6h is EXPERIMENTAL
        
        # Enable an experimental feature
        ml_config.enable_experimental_feature("outdoor_temp_avg_6h")
        ml_config.save()
        
        # Sync again
        sync_stats_config_with_features()
        
        # Verify that the experimental stat is now enabled
        stats_config = get_feature_stats_config()
        outdoor_stats = stats_config.get_sensor_config("outdoor_temp").enabled_stats
        self.assertIn(StatType.AVG_6H, outdoor_stats)


if __name__ == "__main__":
    unittest.main()
