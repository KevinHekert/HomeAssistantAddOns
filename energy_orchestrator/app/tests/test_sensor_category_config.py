"""
Tests for the sensor category configuration module.

Tests the sensor category configuration with Core/Experimental categories.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from db.sensor_category_config import (
    SensorType,
    SensorDefinition,
    SensorConfig,
    SensorCategoryConfiguration,
    CORE_SENSORS,
    EXPERIMENTAL_SENSORS,
    get_all_sensor_definitions,
    get_sensor_definition,
    get_sensor_category_config,
    reload_sensor_category_config,
    get_configured_sensor_entities,
    get_sensor_entity_id,
)


class TestSensorType:
    """Tests for SensorType enum."""

    def test_weather_type(self):
        """Weather type should be 'weather'."""
        assert SensorType.WEATHER.value == "weather"

    def test_indoor_type(self):
        """Indoor type should be 'indoor'."""
        assert SensorType.INDOOR.value == "indoor"

    def test_heating_type(self):
        """Heating type should be 'heating'."""
        assert SensorType.HEATING.value == "heating"

    def test_usage_type(self):
        """Usage type should be 'usage'."""
        assert SensorType.USAGE.value == "usage"


class TestSensorDefinition:
    """Tests for SensorDefinition dataclass."""

    def test_core_sensor_definition(self):
        """Core sensor definition should have is_core=True."""
        sensor = SensorDefinition(
            category_name="test",
            display_name="Test Sensor",
            sensor_type=SensorType.WEATHER,
            description="Test description",
            unit="°C",
            is_core=True,
            env_var="TEST_ENTITY_ID",
            default_entity_id="sensor.test",
        )
        assert sensor.is_core is True
        assert sensor.category_name == "test"

    def test_experimental_sensor_definition(self):
        """Experimental sensor definition should have is_core=False."""
        sensor = SensorDefinition(
            category_name="test_exp",
            display_name="Test Experimental",
            sensor_type=SensorType.HEATING,
            description="Test experimental",
            unit="kWh",
            is_core=False,
            env_var="TEST_EXP_ENTITY_ID",
            default_entity_id="sensor.test_exp",
        )
        assert sensor.is_core is False

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        sensor = SensorDefinition(
            category_name="test",
            display_name="Test",
            sensor_type=SensorType.WEATHER,
            description="Desc",
            unit="°C",
            is_core=True,
            env_var="TEST_ID",
            default_entity_id="sensor.test",
        )
        result = sensor.to_dict()
        assert result["category_name"] == "test"
        assert result["sensor_type"] == "weather"
        assert result["is_core"] is True


class TestSensorConfig:
    """Tests for SensorConfig dataclass."""

    def test_default_enabled(self):
        """Default enabled should be True."""
        config = SensorConfig(
            category_name="test",
            entity_id="sensor.test",
        )
        assert config.enabled is True

    def test_custom_enabled_false(self):
        """Should accept enabled=False."""
        config = SensorConfig(
            category_name="test",
            entity_id="sensor.test",
            enabled=False,
        )
        assert config.enabled is False

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        config = SensorConfig(
            category_name="test",
            entity_id="sensor.test",
            enabled=True,
        )
        result = config.to_dict()
        assert result["category_name"] == "test"
        assert result["entity_id"] == "sensor.test"
        assert result["enabled"] is True

    def test_from_dict(self):
        """Should create from dictionary correctly."""
        data = {
            "category_name": "test",
            "entity_id": "sensor.test",
            "enabled": False,
        }
        config = SensorConfig.from_dict(data)
        assert config.category_name == "test"
        assert config.entity_id == "sensor.test"
        assert config.enabled is False


class TestCoreSensors:
    """Tests for core sensor definitions."""

    def test_core_sensors_not_empty(self):
        """Core sensors list should not be empty."""
        assert len(CORE_SENSORS) > 0

    def test_all_core_sensors_have_is_core_true(self):
        """All core sensors should have is_core=True."""
        for sensor in CORE_SENSORS:
            assert sensor.is_core is True, f"{sensor.category_name} should be core"

    def test_hp_kwh_total_is_core(self):
        """hp_kwh_total should be a core sensor."""
        names = [s.category_name for s in CORE_SENSORS]
        assert "hp_kwh_total" in names

    def test_outdoor_temp_is_core(self):
        """outdoor_temp should be a core sensor."""
        names = [s.category_name for s in CORE_SENSORS]
        assert "outdoor_temp" in names

    def test_indoor_temp_is_core(self):
        """indoor_temp should be a core sensor."""
        names = [s.category_name for s in CORE_SENSORS]
        assert "indoor_temp" in names

    def test_target_temp_is_core(self):
        """target_temp should be a core sensor."""
        names = [s.category_name for s in CORE_SENSORS]
        assert "target_temp" in names

    def test_wind_is_core(self):
        """wind should be a core sensor."""
        names = [s.category_name for s in CORE_SENSORS]
        assert "wind" in names

    def test_humidity_is_core(self):
        """humidity should be a core sensor."""
        names = [s.category_name for s in CORE_SENSORS]
        assert "humidity" in names


class TestExperimentalSensors:
    """Tests for experimental sensor definitions."""

    def test_experimental_sensors_not_empty(self):
        """Experimental sensors list should not be empty."""
        assert len(EXPERIMENTAL_SENSORS) > 0

    def test_all_experimental_sensors_have_is_core_false(self):
        """All experimental sensors should have is_core=False."""
        for sensor in EXPERIMENTAL_SENSORS:
            assert sensor.is_core is False, f"{sensor.category_name} should not be core"

    def test_pressure_is_experimental(self):
        """pressure should be an experimental sensor."""
        names = [s.category_name for s in EXPERIMENTAL_SENSORS]
        assert "pressure" in names

    def test_flow_temp_is_experimental(self):
        """flow_temp should be an experimental sensor."""
        names = [s.category_name for s in EXPERIMENTAL_SENSORS]
        assert "flow_temp" in names

    def test_dhw_active_is_experimental(self):
        """dhw_active should be an experimental sensor."""
        names = [s.category_name for s in EXPERIMENTAL_SENSORS]
        assert "dhw_active" in names


class TestGetAllSensorDefinitions:
    """Tests for get_all_sensor_definitions function."""

    def test_returns_all_sensors(self):
        """Should return both core and experimental sensors."""
        all_sensors = get_all_sensor_definitions()
        assert len(all_sensors) == len(CORE_SENSORS) + len(EXPERIMENTAL_SENSORS)

    def test_core_sensors_first(self):
        """Core sensors should be first in the list."""
        all_sensors = get_all_sensor_definitions()
        for i, sensor in enumerate(CORE_SENSORS):
            assert all_sensors[i].category_name == sensor.category_name


class TestGetSensorDefinition:
    """Tests for get_sensor_definition function."""

    def test_finds_core_sensor(self):
        """Should find a core sensor by name."""
        sensor = get_sensor_definition("outdoor_temp")
        assert sensor is not None
        assert sensor.category_name == "outdoor_temp"
        assert sensor.is_core is True

    def test_finds_experimental_sensor(self):
        """Should find an experimental sensor by name."""
        sensor = get_sensor_definition("pressure")
        assert sensor is not None
        assert sensor.category_name == "pressure"
        assert sensor.is_core is False

    def test_returns_none_for_unknown(self):
        """Should return None for unknown sensor."""
        sensor = get_sensor_definition("nonexistent")
        assert sensor is None


class TestSensorCategoryConfiguration:
    """Tests for SensorCategoryConfiguration class."""

    def test_default_initialization(self):
        """Default initialization should include all sensors."""
        config = SensorCategoryConfiguration()
        
        # Should have all sensors
        all_defs = get_all_sensor_definitions()
        assert len(config.sensors) == len(all_defs)

    def test_core_sensors_enabled_by_default(self):
        """Core sensors should be enabled by default."""
        config = SensorCategoryConfiguration()
        
        for sensor in CORE_SENSORS:
            sensor_config = config.get_sensor_config(sensor.category_name)
            assert sensor_config is not None
            assert sensor_config.enabled is True

    def test_experimental_sensors_disabled_by_default(self):
        """Experimental sensors should be disabled by default."""
        config = SensorCategoryConfiguration()
        
        for sensor in EXPERIMENTAL_SENSORS:
            sensor_config = config.get_sensor_config(sensor.category_name)
            assert sensor_config is not None
            assert sensor_config.enabled is False

    def test_set_entity_id(self):
        """Should update entity_id for a sensor."""
        config = SensorCategoryConfiguration()
        
        result = config.set_entity_id("outdoor_temp", "sensor.custom_temp")
        assert result is True
        
        sensor_config = config.get_sensor_config("outdoor_temp")
        assert sensor_config.entity_id == "sensor.custom_temp"

    def test_set_entity_id_unknown_sensor(self):
        """Should return False for unknown sensor."""
        config = SensorCategoryConfiguration()
        
        result = config.set_entity_id("nonexistent", "sensor.test")
        assert result is False

    def test_enable_experimental_sensor(self):
        """Should enable an experimental sensor."""
        config = SensorCategoryConfiguration()
        
        result = config.enable_sensor("pressure")
        assert result is True
        
        sensor_config = config.get_sensor_config("pressure")
        assert sensor_config.enabled is True

    def test_cannot_enable_core_sensor(self):
        """Should not be able to toggle core sensor."""
        config = SensorCategoryConfiguration()
        
        # Core sensors are already enabled, but enable should return False
        result = config.enable_sensor("outdoor_temp")
        assert result is False

    def test_disable_experimental_sensor(self):
        """Should disable an experimental sensor."""
        config = SensorCategoryConfiguration()
        config.enable_sensor("pressure")
        
        result = config.disable_sensor("pressure")
        assert result is True
        
        sensor_config = config.get_sensor_config("pressure")
        assert sensor_config.enabled is False

    def test_cannot_disable_core_sensor(self):
        """Should not be able to disable core sensor."""
        config = SensorCategoryConfiguration()
        
        result = config.disable_sensor("outdoor_temp")
        assert result is False
        
        # Should still be enabled
        sensor_config = config.get_sensor_config("outdoor_temp")
        assert sensor_config.enabled is True

    def test_get_enabled_sensors(self):
        """Should return only enabled sensors."""
        config = SensorCategoryConfiguration()
        config.enable_sensor("pressure")
        
        enabled = config.get_enabled_sensors()
        
        # Should have all core + pressure
        enabled_names = [s.category_name for s in enabled]
        assert "outdoor_temp" in enabled_names
        assert "pressure" in enabled_names
        assert "flow_temp" not in enabled_names  # Still disabled

    def test_get_enabled_sensor_entity_ids(self):
        """Should return entity IDs for enabled sensors."""
        config = SensorCategoryConfiguration()
        
        entity_ids = config.get_enabled_sensor_entity_ids()
        
        # Should have all core sensors
        assert len(entity_ids) == len(CORE_SENSORS)

    def test_get_sensors_by_type(self):
        """Should group sensors by type."""
        config = SensorCategoryConfiguration()
        
        grouped = config.get_sensors_by_type()
        
        # Should have multiple types
        assert "weather" in grouped
        assert "indoor" in grouped
        assert "usage" in grouped

    def test_to_dict_and_from_dict(self):
        """Should serialize and deserialize correctly."""
        config = SensorCategoryConfiguration()
        config.enable_sensor("pressure")
        config.set_entity_id("outdoor_temp", "sensor.custom_temp")
        
        data = config.to_dict()
        restored = SensorCategoryConfiguration.from_dict(data)
        
        # Check values were preserved
        assert restored.get_sensor_config("pressure").enabled is True
        assert restored.get_sensor_config("outdoor_temp").entity_id == "sensor.custom_temp"


class TestSensorCategoryConfigurationPersistence:
    """Tests for save/load functionality."""

    def test_save_and_load(self, tmp_path):
        """Should save and load configuration correctly."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            config = SensorCategoryConfiguration()
            config.enable_sensor("pressure")
            config.set_entity_id("outdoor_temp", "sensor.custom_temp")
            
            success = config.save()
            assert success is True
            
            # Load and verify
            loaded = SensorCategoryConfiguration.load()
            assert loaded.get_sensor_config("pressure").enabled is True
            assert loaded.get_sensor_config("outdoor_temp").entity_id == "sensor.custom_temp"

    def test_load_returns_default_when_file_missing(self, tmp_path):
        """Should return defaults when config file doesn't exist."""
        config_file = tmp_path / "nonexistent.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            config = SensorCategoryConfiguration.load()
            
            # Should have default values
            assert len(config.sensors) == len(get_all_sensor_definitions())

    def test_handles_corrupted_json(self, tmp_path):
        """Should return defaults for corrupted JSON."""
        config_file = tmp_path / "sensor_category_config.json"
        config_file.write_text("not valid json {{{")
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            config = SensorCategoryConfiguration.load()
            
            # Should have defaults
            assert len(config.sensors) == len(get_all_sensor_definitions())


class TestMigrationFromEnvVars:
    """Tests for migration from environment variables."""

    def test_migration_sets_entity_ids(self, tmp_path, monkeypatch):
        """Should migrate entity IDs from environment variables."""
        config_file = tmp_path / "sensor_category_config.json"
        
        monkeypatch.setenv("OUTDOOR_TEMP_ENTITY_ID", "sensor.my_outdoor_temp")
        monkeypatch.setenv("HP_KWH_TOTAL_ENTITY_ID", "sensor.my_hp_kwh")
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                config = get_sensor_category_config()
                
                assert config.get_sensor_config("outdoor_temp").entity_id == "sensor.my_outdoor_temp"
                assert config.get_sensor_config("hp_kwh_total").entity_id == "sensor.my_hp_kwh"
                assert config.migrated_from_config_yaml is True

    def test_migration_enables_experimental_with_entity_id(self, tmp_path, monkeypatch):
        """Should enable experimental sensors that have env var set."""
        config_file = tmp_path / "sensor_category_config.json"
        
        monkeypatch.setenv("PRESSURE_ENTITY_ID", "sensor.my_pressure")
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                config = get_sensor_category_config()
                
                # Experimental sensor should be enabled
                assert config.get_sensor_config("pressure").enabled is True
                assert config.get_sensor_config("pressure").entity_id == "sensor.my_pressure"

    def test_migration_only_happens_once(self, tmp_path, monkeypatch):
        """Migration should only happen once."""
        config_file = tmp_path / "sensor_category_config.json"
        
        monkeypatch.setenv("OUTDOOR_TEMP_ENTITY_ID", "sensor.first_value")
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                config1 = get_sensor_category_config()
                assert config1.get_sensor_config("outdoor_temp").entity_id == "sensor.first_value"
        
            # Change env var
            monkeypatch.setenv("OUTDOOR_TEMP_ENTITY_ID", "sensor.second_value")
            
            # Reload - should NOT re-migrate
            config2 = reload_sensor_category_config()
            
            # Should still have first value
            assert config2.get_sensor_config("outdoor_temp").entity_id == "sensor.first_value"


class TestGlobalConfiguration:
    """Tests for global configuration singleton."""

    def test_get_sensor_category_config_returns_same_instance(self, tmp_path):
        """Should return the same instance."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                config1 = get_sensor_category_config()
                config2 = get_sensor_category_config()
                
                assert config1 is config2

    def test_reload_creates_new_instance(self, tmp_path):
        """Reload should create a new instance."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                config1 = get_sensor_category_config()
                config2 = reload_sensor_category_config()
                
                assert config1 is not config2


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_configured_sensor_entities(self, tmp_path):
        """Should return entity IDs for enabled sensors."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                entities = get_configured_sensor_entities()
                
                # Should have core sensors
                assert len(entities) >= len(CORE_SENSORS)

    def test_get_sensor_entity_id_for_enabled_sensor(self, tmp_path):
        """Should return entity ID for enabled sensor."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                entity_id = get_sensor_entity_id("outdoor_temp")
                
                # Should have default value
                assert entity_id is not None
                assert "temperature" in entity_id.lower() or entity_id != ""

    def test_get_sensor_entity_id_for_disabled_sensor(self, tmp_path):
        """Should return None for disabled sensor."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                entity_id = get_sensor_entity_id("pressure")
                
                # Experimental sensors are disabled by default
                assert entity_id is None

    def test_get_sensor_entity_id_for_unknown_sensor(self, tmp_path):
        """Should return None for unknown sensor."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                entity_id = get_sensor_entity_id("nonexistent")
                
                assert entity_id is None
