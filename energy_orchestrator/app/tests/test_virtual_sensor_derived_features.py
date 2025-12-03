"""
Tests for virtual sensor derived features toggle functionality.

This test module verifies that derived features from virtual sensors (e.g., temp_delta_avg_1h)
can be enabled and disabled in the feature configuration.
"""

import pytest

from ml.feature_config import (
    FeatureConfiguration,
)
from db.virtual_sensors import (
    VirtualSensorDefinition,
    VirtualSensorOperation,
    VirtualSensorsConfiguration,
    reset_virtual_sensors_config,
)


class TestVirtualSensorDerivedFeatures:
    """Test that derived features from virtual sensors can be toggled."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test environment before each test and clean up after."""
        # Reset virtual sensors configuration before each test
        reset_virtual_sensors_config()
        
        yield
        
        # Clean up - reset virtual sensors configuration after each test
        reset_virtual_sensors_config()
    
    def _setup_temp_delta_virtual_sensor(self) -> VirtualSensorDefinition:
        """
        Helper method to set up a temp_delta virtual sensor in the global config.
        
        Returns:
            The created VirtualSensorDefinition
        """
        # Create a virtual sensor in the global config
        virtual_config = VirtualSensorsConfiguration()
        temp_delta = VirtualSensorDefinition(
            name="temp_delta",
            display_name="Temperature Delta",
            description="Target - Indoor temperature",
            source_sensor1="target_temp",
            source_sensor2="indoor_temp",
            operation=VirtualSensorOperation.SUBTRACT,
            unit="°C",
            enabled=True,
        )
        virtual_config.add_sensor(temp_delta)
        
        # Make this the global config by setting the module-level variable
        # This is necessary because the feature config methods call get_virtual_sensors_config()
        # which returns the singleton. In a production environment, the singleton would be
        # properly initialized through normal application flow.
        import db.virtual_sensors
        db.virtual_sensors._config = virtual_config
        
        return temp_delta
    
    def test_is_derived_sensor_stat_feature_for_virtual_sensor(self):
        """Test that _is_derived_sensor_stat_feature recognizes virtual sensor features."""
        # Set up virtual sensor using helper
        self._setup_temp_delta_virtual_sensor()
        
        # Get feature config
        feature_config = FeatureConfiguration()
        
        # Test that virtual sensor derived features are recognized
        assert feature_config._is_derived_sensor_stat_feature("temp_delta_avg_1h")
        assert feature_config._is_derived_sensor_stat_feature("temp_delta_avg_6h")
        assert feature_config._is_derived_sensor_stat_feature("temp_delta_avg_24h")
        assert feature_config._is_derived_sensor_stat_feature("temp_delta_avg_7d")
        
        # Test that invalid features are not recognized
        assert not feature_config._is_derived_sensor_stat_feature("temp_delta")
        assert not feature_config._is_derived_sensor_stat_feature("nonexistent_avg_1h")
        assert not feature_config._is_derived_sensor_stat_feature("temp_delta_avg_99h")
    
    def test_enable_virtual_sensor_derived_feature(self):
        """Test enabling a derived feature from a virtual sensor."""
        # Set up virtual sensor using helper
        self._setup_temp_delta_virtual_sensor()
        
        # Get feature config
        feature_config = FeatureConfiguration()
        
        # Enable the derived feature
        feature_name = "temp_delta_avg_1h"
        result = feature_config.enable_feature(feature_name)
        
        # Verify it was enabled
        assert result is True
        assert feature_name in feature_config.experimental_enabled
        assert feature_config.experimental_enabled[feature_name] is True
        
        # Verify it appears in active features
        active_names = feature_config.get_active_feature_names()
        assert feature_name in active_names
    
    def test_disable_virtual_sensor_derived_feature(self):
        """Test disabling a derived feature from a virtual sensor."""
        # Set up virtual sensor using helper
        self._setup_temp_delta_virtual_sensor()
        
        # Get feature config
        feature_config = FeatureConfiguration()
        
        # Enable the derived feature first
        feature_name = "temp_delta_avg_1h"
        feature_config.enable_feature(feature_name)
        assert feature_config.experimental_enabled[feature_name] is True
        
        # Now disable it
        result = feature_config.disable_feature(feature_name)
        
        # Verify it was disabled
        assert result is True
        assert feature_name in feature_config.experimental_enabled
        assert feature_config.experimental_enabled[feature_name] is False
        
        # Verify it doesn't appear in active features
        active_names = feature_config.get_active_feature_names()
        assert feature_name not in active_names
    
    def test_create_derived_feature_metadata_for_virtual_sensor(self):
        """Test that metadata can be created for virtual sensor derived features."""
        # Set up virtual sensor using helper
        self._setup_temp_delta_virtual_sensor()
        
        # Get feature config
        feature_config = FeatureConfiguration()
        
        # Create metadata for the derived feature
        feature_name = "temp_delta_avg_1h"
        metadata = feature_config._create_derived_feature_metadata(feature_name, enabled=True)
        
        # Verify metadata was created
        assert metadata is not None
        assert metadata.name == feature_name
        assert metadata.unit == "°C"
        assert metadata.enabled is True
        assert metadata.is_core is False
        assert "Temperature Delta" in metadata.description or "temp_delta" in metadata.description
        assert metadata.time_window.value == "1h"
    
    def test_virtual_sensor_derived_features_in_all_features(self):
        """Test that enabled virtual sensor derived features appear in get_all_features."""
        # Set up virtual sensor using helper
        self._setup_temp_delta_virtual_sensor()
        
        # Get feature config
        feature_config = FeatureConfiguration()
        
        # Enable multiple derived features
        feature_config.enable_feature("temp_delta_avg_1h")
        feature_config.enable_feature("temp_delta_avg_6h")
        
        # Get all features
        all_features = feature_config.get_all_features()
        feature_names = [f.name for f in all_features]
        
        # Verify the virtual sensor derived features are included
        assert "temp_delta_avg_1h" in feature_names
        assert "temp_delta_avg_6h" in feature_names
        
        # Verify they have correct properties
        temp_delta_1h = next(f for f in all_features if f.name == "temp_delta_avg_1h")
        assert temp_delta_1h.enabled is True
        assert temp_delta_1h.is_core is False
        assert temp_delta_1h.unit == "°C"
