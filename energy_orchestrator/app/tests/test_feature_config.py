"""
Tests for the feature configuration module.

Tests:
1. Feature metadata structure
2. Core and experimental features
3. Feature configuration save/load
4. Timezone handling
5. Feature toggling
"""

import os
import pytest
import tempfile
from datetime import datetime, timezone

from ml.feature_config import (
    FeatureMetadata,
    FeatureConfiguration,
    FeatureCategory,
    TimeWindow,
    CORE_FEATURES,
    EXPERIMENTAL_FEATURES,
    DEFAULT_TIMEZONE,
    get_feature_config,
    reload_feature_config,
    convert_utc_to_local_hour,
    get_feature_metadata_dict,
    get_core_feature_count,
    validate_feature_set,
)
import ml.feature_config as config_module


@pytest.fixture
def temp_config_dir(tmp_path, monkeypatch):
    """Create a temporary directory for configuration storage."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(config_module, "CONFIG_DIR", str(config_dir))
    # Reset the global config to None so it loads fresh
    monkeypatch.setattr(config_module, "_config", None)
    return config_dir


class TestFeatureMetadata:
    """Test the FeatureMetadata dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        feature = FeatureMetadata(
            name="outdoor_temp",
            category=FeatureCategory.WEATHER,
            description="Outdoor temperature",
            unit="°C",
            time_window=TimeWindow.NONE,
            is_core=True,
            enabled=True,
        )
        
        result = feature.to_dict()
        
        assert result["name"] == "outdoor_temp"
        assert result["category"] == "weather"
        assert result["description"] == "Outdoor temperature"
        assert result["unit"] == "°C"
        assert result["time_window"] == "none"
        assert result["is_core"] is True
        assert result["enabled"] is True
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "wind",
            "category": "weather",
            "description": "Wind speed",
            "unit": "m/s",
            "time_window": "none",
            "is_core": True,
            "enabled": True,
        }
        
        feature = FeatureMetadata.from_dict(data)
        
        assert feature.name == "wind"
        assert feature.category == FeatureCategory.WEATHER
        assert feature.description == "Wind speed"
        assert feature.unit == "m/s"
        assert feature.time_window == TimeWindow.NONE
        assert feature.is_core is True
        assert feature.enabled is True


class TestCoreFeatures:
    """Test the core baseline features definition."""
    
    def test_core_features_count(self):
        """There should be exactly 13 core baseline features."""
        assert len(CORE_FEATURES) == 13
    
    def test_core_feature_names(self):
        """Core features should have the expected names."""
        expected_names = {
            "outdoor_temp",
            "outdoor_temp_avg_24h",
            "wind",
            "humidity",
            "indoor_temp",
            "indoor_temp_avg_24h",
            "target_temp",
            "target_temp_avg_6h",
            "heating_kwh_last_1h",
            "heating_kwh_last_6h",
            "heating_kwh_last_24h",
            "hour_of_day",
            "delta_target_indoor",
        }
        
        actual_names = {f.name for f in CORE_FEATURES}
        
        assert actual_names == expected_names
    
    def test_all_core_features_are_marked_core(self):
        """All core features should have is_core=True."""
        for feature in CORE_FEATURES:
            assert feature.is_core is True, f"Feature {feature.name} is not marked as core"
    
    def test_all_core_features_are_enabled(self):
        """All core features should have enabled=True."""
        for feature in CORE_FEATURES:
            assert feature.enabled is True, f"Feature {feature.name} is not enabled"
    
    def test_core_features_have_categories(self):
        """Core features should be grouped by categories."""
        categories = {f.category for f in CORE_FEATURES}
        
        assert FeatureCategory.WEATHER in categories
        assert FeatureCategory.INDOOR in categories
        assert FeatureCategory.CONTROL in categories
        assert FeatureCategory.USAGE in categories
        assert FeatureCategory.TIME in categories
    
    def test_wind_and_humidity_in_baseline(self):
        """Wind and humidity should always be in baseline."""
        names = {f.name for f in CORE_FEATURES}
        assert "wind" in names
        assert "humidity" in names


class TestExperimentalFeatures:
    """Test the experimental features definition."""
    
    def test_experimental_features_not_core(self):
        """All experimental features should have is_core=False."""
        for feature in EXPERIMENTAL_FEATURES:
            assert feature.is_core is False, f"Feature {feature.name} is marked as core"
    
    def test_experimental_features_disabled_by_default(self):
        """All experimental features should be disabled by default."""
        for feature in EXPERIMENTAL_FEATURES:
            assert feature.enabled is False, f"Feature {feature.name} is enabled by default"
    
    def test_pressure_is_experimental(self):
        """Pressure should be experimental, not core."""
        experimental_names = {f.name for f in EXPERIMENTAL_FEATURES}
        core_names = {f.name for f in CORE_FEATURES}
        
        assert "pressure" in experimental_names
        assert "pressure" not in core_names
    
    def test_day_of_week_is_experimental(self):
        """day_of_week should be experimental."""
        experimental_names = {f.name for f in EXPERIMENTAL_FEATURES}
        assert "day_of_week" in experimental_names
    
    def test_is_weekend_is_experimental(self):
        """is_weekend should be experimental."""
        experimental_names = {f.name for f in EXPERIMENTAL_FEATURES}
        assert "is_weekend" in experimental_names


class TestFeatureConfiguration:
    """Test the FeatureConfiguration class."""
    
    def test_default_configuration(self, temp_config_dir):
        """Default configuration has correct defaults."""
        config = FeatureConfiguration()
        
        assert config.timezone == DEFAULT_TIMEZONE
        assert len(config.experimental_enabled) > 0
    
    def test_get_all_features(self, temp_config_dir):
        """get_all_features returns core + experimental."""
        config = FeatureConfiguration()
        
        all_features = config.get_all_features()
        
        assert len(all_features) == len(CORE_FEATURES) + len(EXPERIMENTAL_FEATURES)
    
    def test_get_active_features_only_core_by_default(self, temp_config_dir):
        """By default, only core features are active."""
        config = FeatureConfiguration()
        
        active_features = config.get_active_features()
        
        # Only core features should be active
        assert len(active_features) == len(CORE_FEATURES)
        for f in active_features:
            assert f.is_core is True
    
    def test_get_active_feature_names(self, temp_config_dir):
        """get_active_feature_names returns correct names."""
        config = FeatureConfiguration()
        
        names = config.get_active_feature_names()
        core_names = {f.name for f in CORE_FEATURES}
        
        assert set(names) == core_names
    
    def test_enable_experimental_feature(self, temp_config_dir):
        """Experimental feature can be enabled."""
        config = FeatureConfiguration()
        
        assert "pressure" not in config.get_active_feature_names()
        
        result = config.enable_experimental_feature("pressure")
        
        assert result is True
        assert "pressure" in config.get_active_feature_names()
    
    def test_disable_experimental_feature(self, temp_config_dir):
        """Experimental feature can be disabled."""
        config = FeatureConfiguration()
        config.enable_experimental_feature("pressure")
        
        assert "pressure" in config.get_active_feature_names()
        
        result = config.disable_experimental_feature("pressure")
        
        assert result is True
        assert "pressure" not in config.get_active_feature_names()
    
    def test_cannot_toggle_nonexistent_feature(self, temp_config_dir):
        """Toggling a nonexistent feature returns False."""
        config = FeatureConfiguration()
        
        result = config.enable_experimental_feature("nonexistent_feature")
        
        assert result is False
    
    def test_get_features_by_category(self, temp_config_dir):
        """Features are correctly grouped by category."""
        config = FeatureConfiguration()
        
        grouped = config.get_features_by_category()
        
        assert "weather" in grouped
        assert "indoor" in grouped
        assert "control" in grouped
        assert "usage" in grouped
        assert "time" in grouped
        
        # Check that outdoor_temp is in weather
        weather_names = [f.name for f in grouped["weather"]]
        assert "outdoor_temp" in weather_names
    
    def test_set_valid_timezone(self, temp_config_dir):
        """Valid timezone can be set."""
        config = FeatureConfiguration()
        
        result = config.set_timezone("America/New_York")
        
        assert result is True
        assert config.timezone == "America/New_York"
    
    def test_set_invalid_timezone(self, temp_config_dir):
        """Invalid timezone returns False."""
        config = FeatureConfiguration()
        original_tz = config.timezone
        
        result = config.set_timezone("Invalid/Timezone")
        
        assert result is False
        assert config.timezone == original_tz
    
    def test_save_and_load(self, temp_config_dir):
        """Configuration can be saved and loaded."""
        config = FeatureConfiguration()
        config.set_timezone("America/Los_Angeles")
        config.enable_experimental_feature("pressure")
        
        config.save()
        
        # Load fresh
        loaded = FeatureConfiguration.load()
        
        assert loaded.timezone == "America/Los_Angeles"
        assert "pressure" in loaded.get_active_feature_names()
    
    def test_load_returns_default_when_no_file(self, temp_config_dir):
        """Loading when no file exists returns default config."""
        config = FeatureConfiguration.load()
        
        assert config.timezone == DEFAULT_TIMEZONE
    
    def test_to_dict_and_from_dict(self, temp_config_dir):
        """Configuration can be serialized and deserialized."""
        config = FeatureConfiguration()
        config.set_timezone("Europe/London")
        config.enable_experimental_feature("day_of_week")
        
        data = config.to_dict()
        restored = FeatureConfiguration.from_dict(data)
        
        assert restored.timezone == "Europe/London"
        assert restored.experimental_enabled.get("day_of_week") is True


class TestTimezoneConversion:
    """Test timezone conversion for time features."""
    
    def test_convert_utc_to_local_hour_amsterdam(self, temp_config_dir):
        """UTC timestamp is correctly converted to Amsterdam time."""
        config = FeatureConfiguration()
        config.set_timezone("Europe/Amsterdam")
        
        # Winter time: Amsterdam is UTC+1
        utc_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)
        
        local_hour = convert_utc_to_local_hour(utc_time, config)
        
        # 13:00 UTC = 14:00 CET (Amsterdam winter)
        assert local_hour == 14
    
    def test_convert_utc_to_local_hour_summer_time(self, temp_config_dir):
        """Summer time (DST) is correctly handled."""
        config = FeatureConfiguration()
        config.set_timezone("Europe/Amsterdam")
        
        # Summer time: Amsterdam is UTC+2
        utc_time = datetime(2024, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        local_hour = convert_utc_to_local_hour(utc_time, config)
        
        # 12:00 UTC = 14:00 CEST (Amsterdam summer)
        assert local_hour == 14
    
    def test_convert_naive_timestamp(self, temp_config_dir):
        """Naive timestamp is treated as UTC."""
        config = FeatureConfiguration()
        config.set_timezone("Europe/Amsterdam")
        
        # Winter time
        naive_time = datetime(2024, 1, 15, 13, 0, 0)  # Treated as UTC
        
        local_hour = convert_utc_to_local_hour(naive_time, config)
        
        assert local_hour == 14


class TestGlobalConfiguration:
    """Test global configuration singleton."""
    
    def test_get_feature_config_returns_singleton(self, temp_config_dir):
        """get_feature_config returns the same instance."""
        config1 = get_feature_config()
        config2 = get_feature_config()
        
        assert config1 is config2
    
    def test_reload_feature_config_reloads(self, temp_config_dir):
        """reload_feature_config creates a new instance."""
        config1 = get_feature_config()
        config1.set_timezone("America/Chicago")
        config1.save()
        
        config2 = reload_feature_config()
        
        assert config2.timezone == "America/Chicago"


class TestFeatureMetadataHelpers:
    """Test helper functions for feature metadata."""
    
    def test_get_feature_metadata_dict(self, temp_config_dir):
        """get_feature_metadata_dict returns list of dicts."""
        metadata = get_feature_metadata_dict()
        
        assert isinstance(metadata, list)
        assert len(metadata) == len(CORE_FEATURES) + len(EXPERIMENTAL_FEATURES)
        
        # Check structure of first item
        first = metadata[0]
        assert "name" in first
        assert "category" in first
        assert "description" in first
        assert "unit" in first
        assert "is_core" in first
    
    def test_get_core_feature_count(self):
        """get_core_feature_count returns 13."""
        assert get_core_feature_count() == 13
    
    def test_validate_feature_set_complete(self):
        """Complete feature set is valid."""
        feature_names = [f.name for f in CORE_FEATURES]
        
        is_valid, missing = validate_feature_set(feature_names)
        
        assert is_valid is True
        assert missing == []
    
    def test_validate_feature_set_missing(self):
        """Incomplete feature set returns missing features."""
        feature_names = ["outdoor_temp", "wind"]  # Missing most core features
        
        is_valid, missing = validate_feature_set(feature_names)
        
        assert is_valid is False
        assert len(missing) > 0
        assert "humidity" in missing


class TestFeatureCategories:
    """Test feature category distribution."""
    
    def test_weather_category_features(self, temp_config_dir):
        """Weather category has correct features."""
        config = FeatureConfiguration()
        grouped = config.get_features_by_category()
        
        weather = {f.name for f in grouped["weather"]}
        
        assert "outdoor_temp" in weather
        assert "outdoor_temp_avg_24h" in weather
        assert "wind" in weather
        assert "humidity" in weather
    
    def test_usage_category_features(self, temp_config_dir):
        """Usage category has correct features."""
        config = FeatureConfiguration()
        grouped = config.get_features_by_category()
        
        usage = {f.name for f in grouped["usage"]}
        
        assert "heating_kwh_last_1h" in usage
        assert "heating_kwh_last_6h" in usage
        assert "heating_kwh_last_24h" in usage
    
    def test_control_category_has_delta_target_indoor(self, temp_config_dir):
        """Control category includes delta_target_indoor."""
        config = FeatureConfiguration()
        grouped = config.get_features_by_category()
        
        control = {f.name for f in grouped["control"]}
        
        assert "delta_target_indoor" in control


class TestFeatureVerification:
    """Test feature verification functions."""
    
    def test_categorize_features_raw(self):
        """Raw sensor features are correctly categorized."""
        from ml.feature_config import categorize_features, RAW_SENSOR_FEATURES
        
        raw_features = list(RAW_SENSOR_FEATURES)
        result = categorize_features(raw_features)
        
        assert len(result["raw_sensor_features"]) == len(raw_features)
        assert len(result["calculated_features"]) == 0
    
    def test_categorize_features_calculated(self):
        """Calculated features are correctly categorized."""
        from ml.feature_config import categorize_features
        
        calculated = ["outdoor_temp_avg_24h", "heating_kwh_last_6h", "delta_target_indoor"]
        result = categorize_features(calculated)
        
        assert len(result["raw_sensor_features"]) == 0
        assert len(result["calculated_features"]) == 3
    
    def test_categorize_features_mixed(self):
        """Mixed features are correctly categorized."""
        from ml.feature_config import categorize_features
        
        mixed = ["outdoor_temp", "indoor_temp", "outdoor_temp_avg_24h", "hour_of_day"]
        result = categorize_features(mixed)
        
        assert "outdoor_temp" in result["raw_sensor_features"]
        assert "indoor_temp" in result["raw_sensor_features"]
        assert "outdoor_temp_avg_24h" in result["calculated_features"]
        assert "hour_of_day" in result["calculated_features"]
    
    def test_get_feature_details(self):
        """Feature details include metadata."""
        from ml.feature_config import get_feature_details
        
        features = ["outdoor_temp", "heating_kwh_last_6h"]
        details = get_feature_details(features)
        
        assert len(details) == 2
        
        # Check outdoor_temp details
        outdoor = details[0]
        assert outdoor["name"] == "outdoor_temp"
        assert outdoor["category"] == "weather"
        assert outdoor["is_core"] is True
        assert outdoor["is_calculated"] is False
        
        # Check heating_kwh_last_6h details
        heating = details[1]
        assert heating["name"] == "heating_kwh_last_6h"
        assert heating["category"] == "usage"
        assert heating["is_core"] is True
        assert heating["is_calculated"] is True
    
    def test_verify_model_features_all_present(self):
        """Feature verification passes when all features are present."""
        from ml.feature_config import verify_model_features
        
        model_features = ["outdoor_temp", "wind", "humidity"]
        dataset_features = ["outdoor_temp", "wind", "humidity", "extra_feature"]
        
        result = verify_model_features(model_features, dataset_features)
        
        assert result["verified"] is True
        assert result["feature_count"] == 3
        assert set(result["verified_features"]) == {"outdoor_temp", "wind", "humidity"}
        assert len(result["missing_in_dataset"]) == 0
    
    def test_verify_model_features_missing(self):
        """Feature verification fails when features are missing."""
        from ml.feature_config import verify_model_features
        
        model_features = ["outdoor_temp", "wind", "humidity"]
        dataset_features = ["outdoor_temp", "wind"]  # missing humidity
        
        result = verify_model_features(model_features, dataset_features)
        
        assert result["verified"] is False
        assert "humidity" in result["missing_in_dataset"]
    
    def test_verify_model_features_unused(self):
        """Unused features in dataset are reported."""
        from ml.feature_config import verify_model_features
        
        model_features = ["outdoor_temp"]
        dataset_features = ["outdoor_temp", "wind", "humidity"]
        
        result = verify_model_features(model_features, dataset_features)
        
        assert result["verified"] is True
        assert "wind" in result["unused_in_model"]
        assert "humidity" in result["unused_in_model"]
    
    def test_categorize_features_unknown(self):
        """Unknown features are categorized and reported."""
        from ml.feature_config import categorize_features
        
        features = ["outdoor_temp", "unknown_feature", "another_unknown"]
        result = categorize_features(features)
        
        assert "outdoor_temp" in result["raw_sensor_features"]
        assert "unknown_feature" in result["calculated_features"]
        assert "another_unknown" in result["calculated_features"]
        assert "unknown_features" in result
        assert "unknown_feature" in result["unknown_features"]
        assert "another_unknown" in result["unknown_features"]
