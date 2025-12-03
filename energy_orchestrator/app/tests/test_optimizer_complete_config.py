"""
Tests for complete feature configuration storage and application in optimizer.

Verifies that:
1. Optimizer stores complete feature configuration (all features, not just tested ones)
2. Applying an optimizer result restores ALL feature states
3. Legacy results (without complete_feature_config) still work
"""

import pytest
from unittest.mock import patch
from ml.optimizer import OptimizationResult, apply_best_configuration
from ml.feature_config import get_feature_config, CORE_FEATURES, EXPERIMENTAL_FEATURES
import json


def test_optimization_result_has_complete_feature_config():
    """Test that OptimizationResult can store complete feature configuration."""
    # Create a complete feature config with all features
    complete_config = {
        "outdoor_temp": True,
        "wind": True,
        "humidity": False,  # Disabled
        "indoor_temp": True,
        "pressure": True,
        "delta_target_indoor": False,  # Experimental disabled
        "heating_degree_hours_24h": True,  # Experimental enabled
    }
    
    result = OptimizationResult(
        config_name="test_config",
        model_type="single_step",
        experimental_features={"delta_target_indoor": False, "heating_degree_hours_24h": True},
        complete_feature_config=complete_config,
        val_mape_pct=10.5,
        val_mae_kwh=0.15,
        val_r2=0.85,
        train_samples=100,
        val_samples=20,
        success=True,
    )
    
    assert result.complete_feature_config == complete_config
    assert len(result.complete_feature_config) > len(result.experimental_features)
    assert "outdoor_temp" in result.complete_feature_config
    assert "humidity" in result.complete_feature_config


@patch('ml.feature_config.FeatureConfiguration.save', return_value=True)
def test_apply_complete_feature_config(mock_save):
    """Test that applying a result with complete_feature_config restores all features."""
    # Get initial config
    config = get_feature_config()
    
    # Set some specific states
    config.enable_feature("outdoor_temp")
    config.enable_feature("wind")
    config.disable_feature("humidity")
    config.enable_feature("delta_target_indoor")
    
    # Create result with different complete config
    complete_config = {
        "outdoor_temp": True,
        "wind": False,  # Changed
        "humidity": True,  # Changed
        "delta_target_indoor": False,  # Changed
    }
    
    result = OptimizationResult(
        config_name="test_apply",
        model_type="single_step",
        experimental_features={"delta_target_indoor": False},  # Only this in experimental
        complete_feature_config=complete_config,
        val_mape_pct=10.0,
        val_mae_kwh=0.1,
        val_r2=0.9,
        train_samples=100,
        val_samples=20,
        success=True,
    )
    
    # Apply the configuration
    success = apply_best_configuration(result, enable_two_step=False)
    assert success
    assert mock_save.called
    
    # Reload config and verify ALL features were applied
    config = get_feature_config()
    
    # Check the feature states
    all_features = {**config.core_enabled, **config.experimental_enabled}
    assert all_features.get("outdoor_temp") == True
    assert all_features.get("wind") == False  # Should be changed
    assert all_features.get("humidity") == True  # Should be changed
    assert all_features.get("delta_target_indoor") == False  # Should be changed


@patch('ml.feature_config.FeatureConfiguration.save', return_value=True)
def test_apply_legacy_experimental_features_only(mock_save):
    """Test that legacy results (no complete_feature_config) still work."""
    # Get initial config and set a specific state
    config = get_feature_config()
    config.enable_feature("outdoor_temp")
    config.disable_feature("wind")
    config.enable_feature("delta_target_indoor")
    
    # Create result WITHOUT complete_feature_config (legacy format)
    result = OptimizationResult(
        config_name="legacy_test",
        model_type="single_step",
        experimental_features={"delta_target_indoor": False, "heating_degree_hours_24h": True},
        complete_feature_config=None,  # Legacy format
        val_mape_pct=12.0,
        val_mae_kwh=0.2,
        val_r2=0.8,
        train_samples=100,
        val_samples=20,
        success=True,
    )
    
    # Apply the configuration
    success = apply_best_configuration(result, enable_two_step=False)
    assert success
    assert mock_save.called
    
    # Reload config and verify ONLY experimental features were applied
    config = get_feature_config()
    
    # Check the feature states
    all_features = {**config.core_enabled, **config.experimental_enabled}
    
    # Experimental features should be applied
    assert all_features.get("delta_target_indoor") == False
    assert all_features.get("heating_degree_hours_24h") == True
    # Other features should remain unchanged (outdoor_temp and wind)
    assert all_features.get("outdoor_temp") == True  # Unchanged
    assert all_features.get("wind") == False  # Unchanged


def test_get_complete_feature_state():
    """Test that FeatureConfiguration.get_complete_feature_state() returns all features."""
    config = get_feature_config()
    
    # Enable/disable some features
    config.enable_feature("outdoor_temp")
    config.disable_feature("wind")
    config.enable_feature("delta_target_indoor")
    
    # Get complete state
    complete_state = config.get_complete_feature_state()
    
    # Should contain both core and experimental features
    assert isinstance(complete_state, dict)
    assert len(complete_state) > 0
    
    # Should have our configured features
    assert "outdoor_temp" in complete_state
    assert "wind" in complete_state
    assert "delta_target_indoor" in complete_state
    
    # Verify states match
    assert complete_state["outdoor_temp"] == True
    assert complete_state["wind"] == False
    assert complete_state["delta_target_indoor"] == True


def test_complete_config_includes_all_feature_types():
    """Test that complete config includes both core and experimental features."""
    config = get_feature_config()
    
    # Get complete state
    complete_state = config.get_complete_feature_state()
    
    # Should have at least some core features
    core_feature_names = {f.name for f in CORE_FEATURES}
    for core_name in list(core_feature_names)[:3]:  # Check first 3
        assert core_name in complete_state, f"Core feature {core_name} should be in complete state"
    
    # Should have at least some experimental features
    experimental_feature_names = {f.name for f in EXPERIMENTAL_FEATURES}
    for exp_name in list(experimental_feature_names)[:3]:  # Check first 3
        assert exp_name in complete_state, f"Experimental feature {exp_name} should be in complete state"
