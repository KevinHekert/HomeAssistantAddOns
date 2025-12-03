"""
Tests for optimizer's interaction with feature configuration.

This test suite ensures that the optimizer only modifies feature_config (training)
and does NOT modify feature_stats_config (availability).
"""

import pytest
from pathlib import Path

from db.feature_stats import (
    FeatureStatsConfiguration,
    StatType,
    reload_feature_stats_config,
)
from ml.feature_config import (
    FeatureConfiguration,
    reload_feature_config,
)
from ml.optimizer import (
    apply_best_configuration,
    OptimizationResult,
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


class TestOptimizerFeatureSeparation:
    """Test that optimizer only modifies feature_config, not feature_stats_config."""
    
    def test_optimizer_does_not_modify_feature_stats_config(self, temp_config_dirs):
        """
        Test: Applying optimizer result should NOT modify feature_stats_config.
        
        Scenario (the reported bug):
        1. User enables avg_1h for wind in Sensor Configuration tab
        2. User runs optimizer
        3. Optimizer finds a config that disables wind_avg_1h for training
        4. User applies optimizer result
        5. Bug: User goes to Sensor Configuration tab and sees checkbox still checked
        6. But in Configuration tab, the feature is unchecked for training
        
        Expected: Feature stats config should be unchanged, only training config changes
        """
        # Step 1: User enables stat in Sensor Configuration
        stats_config = FeatureStatsConfiguration()
        stats_config.set_stat_enabled("wind", StatType.AVG_1H, True)
        stats_config.save()
        
        # User also enables it for training initially
        feature_config = FeatureConfiguration()
        feature_config.enable_feature("wind_avg_1h")
        feature_config.save()
        
        # Verify both are enabled
        stats_config_check = reload_feature_stats_config()
        assert StatType.AVG_1H in stats_config_check.get_enabled_stats_for_sensor("wind")
        
        feature_config_check = reload_feature_config()
        assert "wind_avg_1h" in feature_config_check.get_active_feature_names()
        
        # Step 2-3: Optimizer finds a config that disables wind_avg_1h
        # Simulate optimizer result that disables wind_avg_1h for training
        optimizer_result = OptimizationResult(
            config_name="best_config",
            model_type="single_step",
            experimental_features={"wind_avg_1h": False},  # Disabled for training
            complete_feature_config={
                "wind_avg_1h": False,  # Disabled for training
                "outdoor_temp": True,
                "wind": True,
            },
            val_mape_pct=5.5,
            val_mae_kwh=0.8,
            val_r2=0.92,
            train_samples=1000,
            val_samples=200,
            success=True,
        )
        
        # Step 4: Apply optimizer result
        success = apply_best_configuration(optimizer_result, enable_two_step=False)
        assert success, "Optimizer config should apply successfully"
        
        # Step 5-6: Check that feature_stats_config is UNCHANGED
        stats_config_after = reload_feature_stats_config()
        enabled_stats = stats_config_after.get_enabled_stats_for_sensor("wind")
        
        assert StatType.AVG_1H in enabled_stats, \
            "Feature stats config should NOT be modified by optimizer - checkbox should still be checked in Sensor Configuration tab"
        
        # But feature_config should be changed (training disabled)
        feature_config_after = reload_feature_config()
        active_features = feature_config_after.get_active_feature_names()
        
        assert "wind_avg_1h" not in active_features, \
            "Feature config should be modified by optimizer - checkbox unchecked in Configuration tab"
    
    def test_optimizer_applies_derived_features_to_training_only(self, temp_config_dirs):
        """
        Test: Optimizer can enable/disable derived features for training without affecting stats collection.
        
        Scenario:
        1. User has pressure_avg_6h available (enabled in stats config)
        2. Initially NOT enabled for training
        3. Optimizer finds config that ENABLES pressure_avg_6h for training
        4. Apply optimizer result
        5. pressure_avg_6h should now be enabled for training
        6. But stats config should remain unchanged
        """
        # Step 1: Enable stat in Sensor Configuration
        stats_config = FeatureStatsConfiguration()
        stats_config.set_stat_enabled("pressure", StatType.AVG_6H, True)
        stats_config.save()
        
        # Step 2: Initially not enabled for training
        feature_config = FeatureConfiguration()
        # pressure_avg_6h exists in experimental_enabled as False by default
        assert "pressure_avg_6h" not in feature_config.get_active_feature_names()
        
        # Step 3: Optimizer finds config that enables it
        optimizer_result = OptimizationResult(
            config_name="with_pressure_avg",
            model_type="single_step",
            experimental_features={"pressure_avg_6h": True},
            complete_feature_config={"pressure_avg_6h": True},
            val_mape_pct=5.1,
            val_mae_kwh=0.75,
            val_r2=0.93,
            train_samples=1000,
            val_samples=200,
            success=True,
        )
        
        # Step 4: Apply optimizer result
        success = apply_best_configuration(optimizer_result, enable_two_step=False)
        assert success
        
        # Step 5: Check training is enabled
        feature_config_after = reload_feature_config()
        assert "pressure_avg_6h" in feature_config_after.get_active_feature_names(), \
            "Optimizer should enable derived feature for training"
        
        # Step 6: Check stats config unchanged
        stats_config_after = reload_feature_stats_config()
        enabled_stats = stats_config_after.get_enabled_stats_for_sensor("pressure")
        assert StatType.AVG_6H in enabled_stats, \
            "Stats config should remain unchanged"
    
    def test_optimizer_disables_multiple_derived_features(self, temp_config_dirs):
        """
        Test: Optimizer can disable multiple derived features without affecting stats collection.
        
        This tests the scenario where optimizer finds a simpler model is better.
        """
        # Setup: Enable several stats
        stats_config = FeatureStatsConfiguration()
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_1H, True)
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_6H, True)
        stats_config.set_stat_enabled("wind", StatType.AVG_1H, True)
        stats_config.set_stat_enabled("wind", StatType.AVG_24H, True)
        stats_config.save()
        
        # Enable them for training
        feature_config = FeatureConfiguration()
        feature_config.enable_feature("outdoor_temp_avg_1h")
        feature_config.enable_feature("outdoor_temp_avg_6h")
        feature_config.enable_feature("wind_avg_1h")
        feature_config.enable_feature("wind_avg_24h")
        feature_config.save()
        
        # Optimizer finds simpler model is better (disables most derived features)
        optimizer_result = OptimizationResult(
            config_name="simple_model",
            model_type="single_step",
            experimental_features={
                "outdoor_temp_avg_1h": False,
                "outdoor_temp_avg_6h": False,
                "wind_avg_1h": False,
                "wind_avg_24h": False,
            },
            complete_feature_config={
                "outdoor_temp_avg_1h": False,
                "outdoor_temp_avg_6h": False,
                "wind_avg_1h": False,
                "wind_avg_24h": False,
            },
            val_mape_pct=5.2,
            val_mae_kwh=0.78,
            val_r2=0.91,
            train_samples=1000,
            val_samples=200,
            success=True,
        )
        
        # Apply optimizer result
        success = apply_best_configuration(optimizer_result, enable_two_step=False)
        assert success
        
        # Check all disabled for training
        feature_config_after = reload_feature_config()
        active_features = feature_config_after.get_active_feature_names()
        
        assert "outdoor_temp_avg_1h" not in active_features
        assert "outdoor_temp_avg_6h" not in active_features
        assert "wind_avg_1h" not in active_features
        assert "wind_avg_24h" not in active_features
        
        # But all still enabled in stats config
        stats_config_after = reload_feature_stats_config()
        
        outdoor_stats = stats_config_after.get_enabled_stats_for_sensor("outdoor_temp")
        assert StatType.AVG_1H in outdoor_stats
        assert StatType.AVG_6H in outdoor_stats
        
        wind_stats = stats_config_after.get_enabled_stats_for_sensor("wind")
        assert StatType.AVG_1H in wind_stats
        assert StatType.AVG_24H in wind_stats
