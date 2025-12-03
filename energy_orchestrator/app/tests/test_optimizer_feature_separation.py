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
        Test: Applying optimizer result NOW DOES enable stats for derived features.
        
        Scenario (updated behavior):
        1. User enables avg_1h for wind in Sensor Configuration tab
        2. User runs optimizer
        3. Optimizer finds a config that uses wind_avg_24h (but avg_24h not enabled in stats)
        4. User applies optimizer result from top 30 list
        5. Expected: avg_24h gets auto-enabled in stats config, wind_avg_1h stays enabled
        6. Both features are now available and the one used by optimizer is enabled for training
        
        This prevents confusion where optimizer enables a feature for training but user
        can't see it because the stat wasn't enabled in Sensor Configuration.
        """
        # Step 1: User enables avg_1h for wind in Sensor Configuration
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
        
        # Step 2-3: Optimizer finds a config that uses wind_avg_24h (NOT wind_avg_1h)
        # Note: avg_24h is NOT enabled in stats config yet
        optimizer_result = OptimizationResult(
            config_name="best_config",
            model_type="single_step",
            experimental_features={"wind_avg_24h": True, "wind_avg_1h": False},
            complete_feature_config={
                "wind_avg_24h": True,  # Enabled for training
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
        
        # Step 4: Apply optimizer result (user clicks Apply button)
        success = apply_best_configuration(optimizer_result, enable_two_step=False)
        assert success, "Optimizer config should apply successfully"
        
        # Step 5-6: Check that avg_24h was AUTO-ENABLED in stats config
        stats_config_after = reload_feature_stats_config()
        enabled_stats = stats_config_after.get_enabled_stats_for_sensor("wind")
        
        assert StatType.AVG_24H in enabled_stats, \
            "avg_24h should be auto-enabled in stats config when optimizer applies it"
        
        # avg_1h should still be enabled (not removed)
        assert StatType.AVG_1H in enabled_stats, \
            "Previously enabled stats should remain enabled"
        
        # But feature_config should show wind_avg_24h enabled, wind_avg_1h disabled
        feature_config_after = reload_feature_config()
        active_features = feature_config_after.get_active_feature_names()
        
        assert "wind_avg_24h" in active_features, \
            "wind_avg_24h should be enabled for training"
        assert "wind_avg_1h" not in active_features, \
            "wind_avg_1h should be disabled for training by optimizer"
    
    def test_optimizer_applies_derived_features_to_training_only(self, temp_config_dirs):
        """
        Test: Optimizer auto-enables stats when applying derived features for training.
        
        Scenario:
        1. User has NOT enabled pressure_avg_6h in stats config
        2. Initially NOT enabled for training
        3. Optimizer finds config that ENABLES pressure_avg_6h for training
        4. Apply optimizer result (user clicks Apply button)
        5. pressure_avg_6h should now be enabled for training
        6. AND avg_6h stat should be auto-enabled in stats config
        """
        # Step 1: pressure_avg_6h NOT enabled in stats config
        stats_config = FeatureStatsConfiguration()
        # Explicitly disable it
        stats_config.set_stat_enabled("pressure", StatType.AVG_6H, False)
        stats_config.save()
        
        # Step 2: Initially not enabled for training
        feature_config = FeatureConfiguration()
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
        
        # Step 6: Check stats config was AUTO-ENABLED
        stats_config_after = reload_feature_stats_config()
        enabled_stats = stats_config_after.get_enabled_stats_for_sensor("pressure")
        assert StatType.AVG_6H in enabled_stats, \
            "Stats config should be auto-enabled when optimizer applies derived feature"
    
    def test_optimizer_enables_multiple_derived_features_and_stats(self, temp_config_dirs):
        """
        Test: Optimizer auto-enables stats for multiple derived features when applying configuration.
        
        This tests the scenario where optimizer finds a model with multiple derived features
        and ensures all required stats are enabled.
        """
        # Setup: No stats enabled initially
        stats_config = FeatureStatsConfiguration()
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_1H, False)
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_6H, False)
        stats_config.set_stat_enabled("wind", StatType.AVG_1H, False)
        stats_config.set_stat_enabled("wind", StatType.AVG_24H, False)
        stats_config.save()
        
        # Optimizer finds model that uses multiple derived features
        optimizer_result = OptimizationResult(
            config_name="multi_feature_model",
            model_type="single_step",
            experimental_features={
                "outdoor_temp_avg_1h": True,
                "outdoor_temp_avg_6h": True,
                "wind_avg_1h": True,
                "wind_avg_24h": True,
            },
            complete_feature_config={
                "outdoor_temp_avg_1h": True,
                "outdoor_temp_avg_6h": True,
                "wind_avg_1h": True,
                "wind_avg_24h": True,
            },
            val_mape_pct=5.0,
            val_mae_kwh=0.72,
            val_r2=0.94,
            train_samples=1000,
            val_samples=200,
            success=True,
        )
        
        # Apply optimizer result
        success = apply_best_configuration(optimizer_result, enable_two_step=False)
        assert success
        
        # Check all enabled for training
        feature_config_after = reload_feature_config()
        active_features = feature_config_after.get_active_feature_names()
        
        assert "outdoor_temp_avg_1h" in active_features
        assert "outdoor_temp_avg_6h" in active_features
        assert "wind_avg_1h" in active_features
        assert "wind_avg_24h" in active_features
        
        # Check all stats auto-enabled
        stats_config_after = reload_feature_stats_config()
        
        outdoor_stats = stats_config_after.get_enabled_stats_for_sensor("outdoor_temp")
        assert StatType.AVG_1H in outdoor_stats, "outdoor_temp avg_1h should be auto-enabled"
        assert StatType.AVG_6H in outdoor_stats, "outdoor_temp avg_6h should be auto-enabled"
        
        wind_stats = stats_config_after.get_enabled_stats_for_sensor("wind")
        assert StatType.AVG_1H in wind_stats, "wind avg_1h should be auto-enabled"
        assert StatType.AVG_24H in wind_stats, "wind avg_24h should be auto-enabled"
