"""
Integration tests for sensor cards and derived features.

These tests verify that the sensor cards API correctly reflects the
feature stats configuration and feature configuration.
"""

import pytest
import json
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
from db.sensor_category_config import (
    get_sensor_category_config,
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


def simulate_get_sensor_feature_cards():
    """
    Test helper that simulates the logic from app.py's get_sensor_feature_cards endpoint.
    
    This duplicates the production logic to enable isolated integration testing without
    requiring a full Flask app context. The duplication is intentional to:
    1. Test the integration logic in isolation
    2. Avoid Flask app setup complexity in unit tests
    3. Enable faster test execution
    
    This tests the integration between feature_stats_config and feature_config
    in determining which features appear in sensor cards.
    """
    from ml.feature_config import reload_feature_config
    from db.feature_stats import reload_feature_stats_config
    from db.sensor_category_config import get_all_sensor_definitions, get_sensor_category_config
    
    # Reload to get latest configuration from disk
    feature_config = reload_feature_config()
    feature_stats_conf = reload_feature_stats_config()
    
    all_features = feature_config.get_all_features()
    
    sensor_category_conf = get_sensor_category_config()
    all_sensor_defs = get_all_sensor_definitions()
    
    # Build sensor metadata
    raw_sensors = {}
    for sensor_def in all_sensor_defs:
        sensor_config = sensor_category_conf.get_sensor_config(sensor_def.category_name)
        if sensor_config:
            raw_sensors[sensor_def.category_name] = {
                "display_name": sensor_def.display_name,
                "unit": sensor_config.unit if sensor_config.unit else sensor_def.unit,
                "type": sensor_def.sensor_type.value,
            }
    
    # Build feature map
    feature_map = {f.name: f for f in all_features}
    
    # Group features by sensor
    sensor_features = {}
    for sensor_name in raw_sensors.keys():
        sensor_features[sensor_name] = []
    
    # Add base sensor features and stats
    for sensor_name in raw_sensors.keys():
        sensor_info = raw_sensors[sensor_name]
        
        # Add base sensor feature if it exists
        if sensor_name in feature_map:
            f = feature_map[sensor_name]
            sensor_features[sensor_name].append({
                "name": f.name,
                "is_core": f.is_core,
                "enabled": f.enabled,
            })
        
        # Get enabled stats from feature_stats_config
        enabled_stats = feature_stats_conf.get_enabled_stats_for_sensor(sensor_name)
        
        # Add features for each enabled statistic
        for stat_type in enabled_stats:
            stat_feature_name = f"{sensor_name}_{stat_type.value}"
            
            if stat_feature_name in feature_map:
                f = feature_map[stat_feature_name]
                sensor_features[sensor_name].append({
                    "name": f.name,
                    "is_core": f.is_core,
                    "enabled": f.enabled,  # enabled in feature_config for training
                })
            else:
                sensor_features[sensor_name].append({
                    "name": stat_feature_name,
                    "is_core": False,
                    "enabled": False,  # not in feature_config yet
                })
    
    # Build sensor cards
    sensor_cards = []
    for sensor_name, features in sensor_features.items():
        if features:
            sensor_cards.append({
                "sensor_name": sensor_name,
                "features": features,
            })
    
    return sensor_cards


class TestSensorCardsIntegration:
    """Test that sensor cards correctly integrate stats config and feature config."""
    
    def test_sensor_card_shows_features_based_on_stats_config(self, temp_config_dirs):
        """
        Test: Sensor cards should only show features that are enabled in stats config.
        
        Scenario:
        1. Enable avg_1h for outdoor_temp in stats config
        2. Sensor card for outdoor_temp should show outdoor_temp_avg_1h
        3. Disable avg_1h for outdoor_temp in stats config
        4. Sensor card for outdoor_temp should NOT show outdoor_temp_avg_1h
        """
        stats_config = FeatureStatsConfiguration()
        
        # Enable avg_1h for outdoor_temp
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_1H, True)
        stats_config.save()
        
        # Get sensor cards
        sensor_cards = simulate_get_sensor_feature_cards()
        
        # Find outdoor_temp card
        outdoor_temp_card = next((card for card in sensor_cards if card["sensor_name"] == "outdoor_temp"), None)
        assert outdoor_temp_card is not None
        
        # Check that outdoor_temp_avg_1h is in the features list
        feature_names = [f["name"] for f in outdoor_temp_card["features"]]
        assert "outdoor_temp_avg_1h" in feature_names, "outdoor_temp_avg_1h should be in sensor card when stat is enabled"
        
        # Now disable avg_1h
        stats_config.set_stat_enabled("outdoor_temp", StatType.AVG_1H, False)
        stats_config.save()
        
        # Get sensor cards again
        sensor_cards = simulate_get_sensor_feature_cards()
        outdoor_temp_card = next((card for card in sensor_cards if card["sensor_name"] == "outdoor_temp"), None)
        
        # Check that outdoor_temp_avg_1h is NOT in the features list
        feature_names = [f["name"] for f in outdoor_temp_card["features"]]
        assert "outdoor_temp_avg_1h" not in feature_names, "outdoor_temp_avg_1h should NOT be in sensor card when stat is disabled"
    
    def test_sensor_card_checkbox_reflects_feature_config(self, temp_config_dirs):
        """
        Test: Checkbox state in sensor cards should reflect feature_config (training), not stats_config.
        
        Scenario:
        1. Enable avg_6h for pressure in stats config (makes it available)
        2. Sensor card shows pressure_avg_6h with checkbox unchecked (not enabled for training)
        3. Enable pressure_avg_6h in feature config (for training)
        4. Sensor card shows pressure_avg_6h with checkbox checked
        
        Note: Using pressure sensor because it's experimental and not enabled by default,
        making the test behavior clearer than using core features like indoor_temp.
        """
        stats_config = FeatureStatsConfiguration()
        feature_config = FeatureConfiguration()
        
        # Enable avg_6h for pressure in stats config
        stats_config.set_stat_enabled("pressure", StatType.AVG_6H, True)
        stats_config.save()
        
        sensor_cards = simulate_get_sensor_feature_cards()
        pressure_card = next((card for card in sensor_cards if card["sensor_name"] == "pressure"), None)
        
        # Find pressure_avg_6h feature
        pressure_avg_6h = next((f for f in pressure_card["features"] if f["name"] == "pressure_avg_6h"), None)
        assert pressure_avg_6h is not None
        assert pressure_avg_6h["enabled"] is False, "Should not be enabled for training initially"
        
        # Enable it for training
        feature_config.enable_feature("pressure_avg_6h")
        feature_config.save()
        
        # Get sensor cards again
        sensor_cards = simulate_get_sensor_feature_cards()
        pressure_card = next((card for card in sensor_cards if card["sensor_name"] == "pressure"), None)
        pressure_avg_6h = next((f for f in pressure_card["features"] if f["name"] == "pressure_avg_6h"), None)
        
        assert pressure_avg_6h["enabled"] is True, "Should be enabled for training after feature_config change"
    
    def test_toggling_training_does_not_remove_from_sensor_cards(self, temp_config_dirs):
        """
        Test: Toggling a feature off for training should not remove it from sensor cards.
        
        Scenario:
        1. Enable avg_24h for wind in stats config
        2. Enable wind_avg_24h for training in feature config
        3. Sensor card shows wind_avg_24h with checkbox checked
        4. Disable wind_avg_24h for training in feature config
        5. Sensor card still shows wind_avg_24h, but checkbox is unchecked
        """
        stats_config = FeatureStatsConfiguration()
        feature_config = FeatureConfiguration()
        
        # Enable avg_24h in stats config
        stats_config.set_stat_enabled("wind", StatType.AVG_24H, True)
        stats_config.save()
        
        # Enable for training
        feature_config.enable_feature("wind_avg_24h")
        feature_config.save()
        
        # Get sensor cards
        sensor_cards = simulate_get_sensor_feature_cards()
        wind_card = next((card for card in sensor_cards if card["sensor_name"] == "wind"), None)
        wind_avg_24h = next((f for f in wind_card["features"] if f["name"] == "wind_avg_24h"), None)
        
        assert wind_avg_24h is not None
        assert wind_avg_24h["enabled"] is True
        
        # Disable for training
        feature_config.disable_feature("wind_avg_24h")
        feature_config.save()
        
        # Get sensor cards again
        sensor_cards = simulate_get_sensor_feature_cards()
        wind_card = next((card for card in sensor_cards if card["sensor_name"] == "wind"), None)
        wind_avg_24h = next((f for f in wind_card["features"] if f["name"] == "wind_avg_24h"), None)
        
        # Feature should still be there (because it's still in stats config)
        assert wind_avg_24h is not None, "Feature should still be in sensor card"
        # But not enabled for training
        assert wind_avg_24h["enabled"] is False, "Feature should not be enabled for training"
    
    def test_complete_workflow_with_sensor_cards(self, temp_config_dirs):
        """
        Test: Complete workflow demonstrating the separation of concerns.
        
        Complete scenario:
        1. User goes to Sensor Configuration tab
        2. User enables avg_1h and avg_6h for humidity
        3. Configuration tab sensor cards now show humidity_avg_1h and humidity_avg_6h
        4. Both have checkboxes unchecked (not used for training)
        5. User checks humidity_avg_1h checkbox in Configuration tab
        6. humidity_avg_1h is now used for training
        7. User goes back to Sensor Configuration tab
        8. User unchecks avg_6h for humidity
        9. Configuration tab no longer shows humidity_avg_6h
        10. But humidity_avg_1h is still there and still checked (used for training)
        """
        stats_config = FeatureStatsConfiguration()
        feature_config = FeatureConfiguration()
        
        # Step 1-2: Enable stats in Sensor Configuration
        stats_config.set_stat_enabled("humidity", StatType.AVG_1H, True)
        stats_config.set_stat_enabled("humidity", StatType.AVG_6H, True)
        stats_config.save()
        
        # Step 3: Get sensor cards
        sensor_cards = simulate_get_sensor_feature_cards()
        humidity_card = next((card for card in sensor_cards if card["sensor_name"] == "humidity"), None)
        feature_names = [f["name"] for f in humidity_card["features"]]
        
        assert "humidity_avg_1h" in feature_names
        assert "humidity_avg_6h" in feature_names
        
        # Step 4: Check that both are not enabled for training
        humidity_avg_1h = next((f for f in humidity_card["features"] if f["name"] == "humidity_avg_1h"), None)
        humidity_avg_6h = next((f for f in humidity_card["features"] if f["name"] == "humidity_avg_6h"), None)
        
        assert humidity_avg_1h["enabled"] is False
        assert humidity_avg_6h["enabled"] is False
        
        # Step 5: Enable humidity_avg_1h for training
        feature_config.enable_feature("humidity_avg_1h")
        feature_config.save()
        
        # Step 6: Verify it's used for training
        sensor_cards = simulate_get_sensor_feature_cards()
        humidity_card = next((card for card in sensor_cards if card["sensor_name"] == "humidity"), None)
        humidity_avg_1h = next((f for f in humidity_card["features"] if f["name"] == "humidity_avg_1h"), None)
        
        assert humidity_avg_1h["enabled"] is True
        
        # Step 7-8: Disable avg_6h in Sensor Configuration
        stats_config.set_stat_enabled("humidity", StatType.AVG_6H, False)
        stats_config.save()
        
        # Step 9: Get sensor cards again
        sensor_cards = simulate_get_sensor_feature_cards()
        humidity_card = next((card for card in sensor_cards if card["sensor_name"] == "humidity"), None)
        feature_names = [f["name"] for f in humidity_card["features"]]
        
        # humidity_avg_6h should not be in sensor cards anymore
        assert "humidity_avg_6h" not in feature_names
        
        # Step 10: humidity_avg_1h should still be there and still enabled for training
        assert "humidity_avg_1h" in feature_names
        humidity_avg_1h = next((f for f in humidity_card["features"] if f["name"] == "humidity_avg_1h"), None)
        assert humidity_avg_1h["enabled"] is True, "Training configuration should be independent of stats changes"
