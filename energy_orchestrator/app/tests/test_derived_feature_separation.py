"""
Tests for the separation between feature stats configuration and feature configuration.

This test suite ensures that:
1. Sensor Configuration tab (feature_stats_config) controls which derived features are AVAILABLE
2. Configuration tab (feature_config) controls which features are used for MODEL TRAINING
3. Changes in one do NOT affect the other inappropriately
"""

import pytest
import json
from pathlib import Path

from db.feature_stats import (
    get_feature_stats_config,
    FeatureStatsConfiguration,
    StatType,
)
from ml.feature_config import (
    get_feature_config,
    FeatureConfiguration,
    reload_feature_config,
)
import ml.feature_config as config_module
import db.feature_stats as stats_module


@pytest.fixture
def temp_config_dirs(tmp_path, monkeypatch):
    """Create temporary directories for both configurations."""
    # Setup feature_config directory
    feature_config_dir = tmp_path / "feature_config"
    feature_config_dir.mkdir()
    monkeypatch.setattr(config_module, "CONFIG_DIR", str(feature_config_dir))
    monkeypatch.setattr(config_module, "_config", None)
    
    # Setup feature_stats directory
    stats_config_dir = tmp_path / "stats_config"
    stats_config_dir.mkdir()
    monkeypatch.setattr(stats_module, "DATA_DIR", stats_config_dir)
    monkeypatch.setattr(stats_module, "FEATURE_STATS_CONFIG_FILE", stats_config_dir / "feature_stats_config.json")
    monkeypatch.setattr(stats_module, "_config", None)
    
    return {
        "feature_config_dir": feature_config_dir,
        "stats_config_dir": stats_config_dir,
    }


class TestDerivedFeatureSeparation:
    """Test that feature stats config and feature config are properly separated."""
    
    def test_enabling_feature_stat_does_not_enable_feature_for_training(self, temp_config_dirs):
        """
        Test: Enabling a stat in Sensor Configuration should NOT automatically enable it for training.
        
        User story:
        - User goes to Sensor Configuration tab
        - User enables avg_1h for wind (checkbox in Feature Stats Configuration)
        - This makes wind_avg_1h AVAILABLE as a derived feature
        - But it should NOT automatically enable it for model training
        
        Note: Using wind sensor because wind_avg_1h is not a core feature,
        making the test behavior clearer than using features enabled by default.
        """
        # Get fresh configurations
        stats_config = FeatureStatsConfiguration()
        feature_config = FeatureConfiguration()
        
        # Enable the stat in feature stats config
        stats_config.set_stat_enabled("wind", StatType.AVG_1H, True)
        stats_config.save()
        
        # Check that the feature is available (can be queried)
        enabled_stats = stats_config.get_enabled_stats_for_sensor("wind")
        assert StatType.AVG_1H in enabled_stats
        
        # Check that the feature config INDEPENDENTLY controls training usage
        # The feature should exist in feature config but its enabled state is separate
        feature_config = FeatureConfiguration.load()
        
        # The feature might not be in experimental_enabled yet
        # When we query all features, it should show as not enabled for training
        all_features = feature_config.get_all_features()
        wind_avg_1h_features = [f for f in all_features if f.name == "wind_avg_1h"]
        
        # The feature should be recognized as a valid derived feature
        assert feature_config._is_derived_sensor_stat_feature("wind_avg_1h")
        
        # But it should not be in active features (used for training) unless explicitly enabled
        active_feature_names = feature_config.get_active_feature_names()
        # Since wind_avg_1h is not a CORE feature and not explicitly enabled, it should not be active
        assert "wind_avg_1h" not in active_feature_names
    
    def test_disabling_feature_stat_removes_from_availability(self, temp_config_dirs):
        """
        Test: Disabling a stat in Sensor Configuration removes it from available features.
        
        User story:
        - User has outdoor_temp_avg_6h enabled in Feature Stats Configuration
        - User disables it in Sensor Configuration tab
        - The feature should no longer be recognized as available
        - Sensor cards in Configuration tab should not show it
        """
        stats_config = FeatureStatsConfiguration()
        feature_config = FeatureConfiguration()
        
        # Enable, then disable a stat
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_6H, True)
        stats_config.save()
        
        # Verify it's enabled
        enabled_stats = stats_config.get_enabled_stats_for_sensor("outdoor_temp")
        assert StatType.AVG_6H in enabled_stats
        
        # Now disable it
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_6H, False)
        stats_config.save()
        
        # Verify it's no longer enabled in stats config
        enabled_stats = stats_config.get_enabled_stats_for_sensor("outdoor_temp")
        assert StatType.AVG_6H not in enabled_stats
        
        # The feature config should still be independent - if it was enabled for training,
        # it stays enabled for training, but it won't be available in sensor cards
        # This is the correct behavior: stats config controls AVAILABILITY, feature config controls USAGE
    
    def test_toggling_feature_training_does_not_affect_stats_config(self, temp_config_dirs):
        """
        Test: Toggling a feature in Configuration tab should only affect training, not stats collection.
        
        User story:
        - User has outdoor_temp_avg_1h available (enabled in Feature Stats Configuration)
        - User toggles it OFF in Configuration tab (Feature Configuration)
        - This should only disable it for model training
        - It should still be collected during resampling (stats config unchanged)
        """
        stats_config = FeatureStatsConfiguration()
        feature_config = FeatureConfiguration()
        
        # Enable stat in stats config (makes it available)
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_1H, True)
        stats_config.save()
        
        # Enable it for training in feature config
        feature_config.enable_feature("outdoor_temp_avg_1h")
        feature_config.save()
        
        # Verify it's enabled for training
        assert "outdoor_temp_avg_1h" in feature_config.get_active_feature_names()
        
        # Now disable it for training
        feature_config.disable_feature("outdoor_temp_avg_1h")
        feature_config.save()
        
        # Verify it's no longer active for training
        assert "outdoor_temp_avg_1h" not in feature_config.get_active_feature_names()
        
        # Verify stats config is UNCHANGED
        stats_config_reloaded = FeatureStatsConfiguration.load()
        enabled_stats = stats_config_reloaded.get_enabled_stats_for_sensor("outdoor_temp")
        assert StatType.AVG_1H in enabled_stats, "Stats config should be unchanged by feature config changes"
    
    def test_sensor_cards_visibility_follows_stats_config(self, temp_config_dirs):
        """
        Test: Sensor cards in Configuration tab should only show features that are available in stats config.
        
        User story:
        - User enables avg_1h for pressure in Sensor Configuration tab
        - Configuration tab should show pressure_avg_1h in the pressure sensor card
        - User disables avg_1h for pressure in Sensor Configuration tab
        - Configuration tab should no longer show pressure_avg_1h in the pressure sensor card
        """
        stats_config = FeatureStatsConfiguration()
        
        # Initially disable all stats for pressure
        for stat_type in StatType:
            stats_config.set_stat_enabled("pressure", stat_type, False)
        stats_config.save()
        
        # Get enabled stats - should be empty
        enabled_stats = stats_config.get_enabled_stats_for_sensor("pressure")
        assert len(enabled_stats) == 0
        
        # Now enable avg_1h
        stats_config.set_stat_enabled("pressure", StatType.AVG_1H, True)
        stats_config.save()
        
        # Get enabled stats - should contain avg_1h
        enabled_stats = stats_config.get_enabled_stats_for_sensor("pressure")
        assert StatType.AVG_1H in enabled_stats
        assert len(enabled_stats) == 1
        
        # This means pressure_avg_1h should be available in sensor cards
        # The actual sensor cards API would use this to determine what to show
    
    def test_configuration_tab_checkbox_only_affects_training(self, temp_config_dirs):
        """
        Test: Checkboxes in Configuration tab should ONLY control model training, not stats collection.
        
        User story:
        - User has outdoor_temp_avg_6h available (enabled in stats config)
        - User sees it in Configuration tab with a checkbox
        - Checkbox state reflects whether it's used for training (feature_config)
        - Toggling checkbox only changes feature_config, not stats_config
        """
        stats_config = FeatureStatsConfiguration()
        feature_config = FeatureConfiguration()
        
        # Make outdoor_temp_avg_6h available
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_6H, True)
        stats_config.save()
        
        # Enable it for training
        feature_config.enable_feature("outdoor_temp_avg_6h")
        feature_config.save()
        
        # Verify it's enabled for training
        assert "outdoor_temp_avg_6h" in feature_config.get_active_feature_names()
        
        # Verify stats config is unchanged
        stats_config_check = FeatureStatsConfiguration.load()
        enabled_stats = stats_config_check.get_enabled_stats_for_sensor("outdoor_temp")
        assert StatType.AVG_6H in enabled_stats
        
        # Disable it for training
        feature_config.disable_feature("outdoor_temp_avg_6h")
        feature_config.save()
        
        # Verify it's no longer active for training
        assert "outdoor_temp_avg_6h" not in feature_config.get_active_feature_names()
        
        # Verify stats config is STILL unchanged
        stats_config_check = FeatureStatsConfiguration.load()
        enabled_stats = stats_config_check.get_enabled_stats_for_sensor("outdoor_temp")
        assert StatType.AVG_6H in enabled_stats, "Toggling training should not affect stats collection"
    
    def test_complete_workflow_scenario(self, temp_config_dirs):
        """
        Test: Complete workflow from enabling stats to training with derived features.
        
        Complete user story:
        1. User goes to Sensor Configuration tab
        2. User enables avg_1h and avg_24h for wind sensor (Feature Stats Configuration)
        3. This makes wind_avg_1h and wind_avg_24h available as derived features
        4. User goes to Configuration tab
        5. User sees wind sensor card with checkboxes for wind_avg_1h and wind_avg_24h
        6. User enables wind_avg_1h for training (checkbox in Configuration tab)
        7. wind_avg_1h is now used for model training
        8. wind_avg_24h is still collected (stats config) but not used for training (feature config)
        """
        stats_config = FeatureStatsConfiguration()
        feature_config = FeatureConfiguration()
        
        # Step 1-2: User enables stats in Sensor Configuration tab
        stats_config.set_stat_enabled("wind", StatType.AVG_1H, True)
        stats_config.set_stat_enabled("wind", StatType.AVG_24H, True)
        stats_config.save()
        
        # Step 3: Verify derived features are available
        enabled_stats = stats_config.get_enabled_stats_for_sensor("wind")
        assert StatType.AVG_1H in enabled_stats
        assert StatType.AVG_24H in enabled_stats
        
        # Verify they are recognized as valid derived features
        assert feature_config._is_derived_sensor_stat_feature("wind_avg_1h")
        assert feature_config._is_derived_sensor_stat_feature("wind_avg_24h")
        
        # Step 4-6: User enables wind_avg_1h for training in Configuration tab
        feature_config.enable_feature("wind_avg_1h")
        feature_config.save()
        
        # Step 7: Verify wind_avg_1h is used for training
        active_features = feature_config.get_active_feature_names()
        assert "wind_avg_1h" in active_features
        
        # Step 8: Verify wind_avg_24h is collected but not used for training
        assert "wind_avg_24h" not in active_features, "Should not be active for training unless explicitly enabled"
        enabled_stats = stats_config.get_enabled_stats_for_sensor("wind")
        assert StatType.AVG_24H in enabled_stats, "Should still be collected during resampling"
    
    def test_sensor_configuration_changes_do_not_affect_feature_config(self, temp_config_dirs):
        """
        Test: Changes in Sensor Configuration tab should not affect feature config checkboxes.
        
        User story:
        - User has pressure_avg_1h enabled for training (checked in Configuration tab)
        - User goes to Sensor Configuration tab
        - User disables avg_1h for pressure (unchecks in Feature Stats Configuration)
        - User goes back to Configuration tab
        - pressure_avg_1h checkbox should still reflect its previous training state
        - (Though the feature may no longer be available/visible)
        """
        stats_config = FeatureStatsConfiguration()
        feature_config = FeatureConfiguration()
        
        # Enable stat and feature for training
        stats_config.set_stat_enabled("pressure", StatType.AVG_1H, True)
        stats_config.save()
        feature_config.enable_feature("pressure_avg_1h")
        feature_config.save()
        
        # Verify it's enabled for training
        assert "pressure_avg_1h" in feature_config.get_active_feature_names()
        
        # User disables the stat in Sensor Configuration tab
        stats_config.set_stat_enabled("pressure", StatType.AVG_1H, False)
        stats_config.save()
        
        # Reload feature config to ensure it's reading from disk
        feature_config_reloaded = FeatureConfiguration.load()
        
        # Feature config should be unchanged - still enabled for training
        # (Even though the stat is no longer collected)
        assert "pressure_avg_1h" in feature_config_reloaded.get_active_feature_names()
        
        # Note: In the UI, this feature might not be visible anymore in sensor cards
        # because it's not available in stats config, but the training configuration
        # remains unchanged. This is correct behavior - they are independent.
