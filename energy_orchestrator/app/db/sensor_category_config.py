"""
Sensor category configuration for energy monitoring.

This module provides flexible sensor configuration with categories:
- Core: Essential sensors that are always required (e.g., hp_kwh_total, outdoor_temp)
- Experimental: Optional sensors that can be enabled/disabled via UI

Each sensor has:
- category_name: The internal category name (e.g., "outdoor_temp", "wind")
- entity_id: The Home Assistant entity ID
- is_core: Whether this sensor is required (cannot be disabled)
- enabled: Whether the sensor is currently enabled (always True for core sensors)
- description: Human-readable description
- unit: Expected unit of measurement

Configuration is stored in a JSON file at /data/sensor_category_config.json.
On first run, values are migrated from environment variables (config.yaml).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

_Logger = logging.getLogger(__name__)

# Configuration file path for persistent sensor config storage
# In Home Assistant add-ons, /data is the persistent data directory
# Validate that DATA_DIR is a safe path (no path traversal)
_data_dir_env = os.environ.get("DATA_DIR", "/data")
# Resolve to absolute path and ensure it's within expected locations
_resolved_data_dir = Path(_data_dir_env).resolve()
# Only allow paths under /data, /tmp, or current working directory (for tests)
_allowed_prefixes = ("/data", "/tmp", str(Path.cwd()))
if not any(_resolved_data_dir.as_posix().startswith(prefix) for prefix in _allowed_prefixes):
    _Logger.warning(
        "DATA_DIR '%s' is not in an allowed location, using default /data",
        _data_dir_env,
    )
    _resolved_data_dir = Path("/data")
DATA_DIR = _resolved_data_dir
SENSOR_CONFIG_FILE_PATH = DATA_DIR / "sensor_category_config.json"


class SensorType(str, Enum):
    """Types of sensors for grouping in the UI."""
    WEATHER = "weather"
    INDOOR = "indoor"
    HEATING = "heating"
    USAGE = "usage"


@dataclass
class SensorDefinition:
    """
    Definition of a sensor with its configuration.
    
    Attributes:
        category_name: Internal category name (e.g., "outdoor_temp", "wind")
        display_name: Human-readable name for UI
        sensor_type: Type of sensor for UI grouping
        description: Human-readable description
        unit: Expected unit of measurement
        is_core: True for essential sensors that cannot be disabled
        env_var: Environment variable name for migration from config.yaml
        default_entity_id: Default entity ID for fresh installations
    """
    category_name: str
    display_name: str
    sensor_type: SensorType
    description: str
    unit: str
    is_core: bool
    env_var: str
    default_entity_id: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "category_name": self.category_name,
            "display_name": self.display_name,
            "sensor_type": self.sensor_type.value,
            "description": self.description,
            "unit": self.unit,
            "is_core": self.is_core,
            "env_var": self.env_var,
            "default_entity_id": self.default_entity_id,
        }


@dataclass
class SensorConfig:
    """
    Configuration for a single sensor instance.
    
    Attributes:
        category_name: Internal category name
        entity_id: The Home Assistant entity ID
        enabled: Whether the sensor is enabled (always True for core sensors)
        unit: Optional unit of measurement for display/override purposes
    """
    category_name: str
    entity_id: str
    enabled: bool = True
    unit: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "category_name": self.category_name,
            "entity_id": self.entity_id,
            "enabled": self.enabled,
            "unit": self.unit,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SensorConfig":
        """Create from dictionary."""
        return cls(
            category_name=data["category_name"],
            entity_id=data.get("entity_id", ""),
            enabled=data.get("enabled", True),
            unit=data.get("unit", ""),
        )


# =============================================================================
# CORE SENSORS (always required, cannot be disabled)
# =============================================================================

CORE_SENSORS: list[SensorDefinition] = [
    SensorDefinition(
        category_name="hp_kwh_total",
        display_name="Heat Pump kWh Total",
        sensor_type=SensorType.USAGE,
        description="Total energy consumption of the heat pump (cumulative kWh). Required for training and predictions.",
        unit="kWh",
        is_core=True,
        env_var="HP_KWH_TOTAL_ENTITY_ID",
        default_entity_id="sensor.extra_total",
    ),
    SensorDefinition(
        category_name="outdoor_temp",
        display_name="Outdoor Temperature",
        sensor_type=SensorType.WEATHER,
        description="Current outdoor temperature. Essential for heating demand prediction.",
        unit="°C",
        is_core=True,
        env_var="OUTDOOR_TEMP_ENTITY_ID",
        default_entity_id="sensor.smile_outdoor_temperature",
    ),
    SensorDefinition(
        category_name="indoor_temp",
        display_name="Indoor Temperature",
        sensor_type=SensorType.INDOOR,
        description="Current indoor temperature. Required for setpoint calculations.",
        unit="°C",
        is_core=True,
        env_var="INDOOR_TEMP_ENTITY_ID",
        default_entity_id="sensor.anna_temperature",
    ),
    SensorDefinition(
        category_name="target_temp",
        display_name="Target Temperature",
        sensor_type=SensorType.INDOOR,
        description="Thermostat setpoint temperature. Required for heating predictions.",
        unit="°C",
        is_core=True,
        env_var="TARGET_TEMP_ENTITY_ID",
        default_entity_id="sensor.anna_setpoint",
    ),
    SensorDefinition(
        category_name="wind",
        display_name="Wind Speed",
        sensor_type=SensorType.WEATHER,
        description="Current wind speed. Affects heat loss and heating demand.",
        unit="m/s",
        is_core=True,
        env_var="WIND_ENTITY_ID",
        default_entity_id="sensor.knmi_windsnelheid",
    ),
    SensorDefinition(
        category_name="humidity",
        display_name="Humidity",
        sensor_type=SensorType.WEATHER,
        description="Current outdoor relative humidity. Affects thermal comfort.",
        unit="%",
        is_core=True,
        env_var="HUMIDITY_ENTITY_ID",
        default_entity_id="sensor.knmi_luchtvochtigheid",
    ),
]


# =============================================================================
# EXPERIMENTAL SENSORS (optional, can be enabled/disabled)
# =============================================================================

EXPERIMENTAL_SENSORS: list[SensorDefinition] = [
    SensorDefinition(
        category_name="pressure",
        display_name="Barometric Pressure",
        sensor_type=SensorType.WEATHER,
        description="Current barometric pressure. May help predict weather-related heating patterns.",
        unit="hPa",
        is_core=False,
        env_var="PRESSURE_ENTITY_ID",
        default_entity_id="sensor.knmi_luchtdruk",
    ),
    SensorDefinition(
        category_name="flow_temp",
        display_name="Flow Temperature",
        sensor_type=SensorType.HEATING,
        description="Heat pump flow/supply water temperature. Experimental - may improve predictions for advanced users.",
        unit="°C",
        is_core=False,
        env_var="FLOW_TEMP_ENTITY_ID",
        default_entity_id="sensor.opentherm_water_temperature",
    ),
    SensorDefinition(
        category_name="return_temp",
        display_name="Return Temperature",
        sensor_type=SensorType.HEATING,
        description="Heat pump return water temperature. Experimental - may improve predictions for advanced users.",
        unit="°C",
        is_core=False,
        env_var="RETURN_TEMP_ENTITY_ID",
        default_entity_id="sensor.opentherm_return_temperature",
    ),
    SensorDefinition(
        category_name="dhw_temp",
        display_name="DHW Temperature",
        sensor_type=SensorType.HEATING,
        description="Domestic hot water temperature. Used to filter DHW cycles from heating data.",
        unit="°C",
        is_core=False,
        env_var="DHW_TEMP_ENTITY_ID",
        default_entity_id="sensor.opentherm_dhw_temperature",
    ),
    SensorDefinition(
        category_name="dhw_active",
        display_name="DHW Active",
        sensor_type=SensorType.HEATING,
        description="Binary sensor indicating if DHW heating is active. Helps exclude DHW from training.",
        unit="on/off",
        is_core=False,
        env_var="DHW_ACTIVE_ENTITY_ID",
        default_entity_id="binary_sensor.dhw_active",
    ),
]


def get_all_sensor_definitions() -> list[SensorDefinition]:
    """Get all sensor definitions (core + experimental)."""
    return CORE_SENSORS + EXPERIMENTAL_SENSORS


def get_sensor_definition(category_name: str) -> Optional[SensorDefinition]:
    """Get a sensor definition by category name."""
    for sensor in get_all_sensor_definitions():
        if sensor.category_name == category_name:
            return sensor
    return None


@dataclass
class SensorCategoryConfiguration:
    """
    Complete sensor category configuration.
    
    Stores:
    - Entity IDs for each sensor category
    - Enabled/disabled state for experimental sensors
    - Migration status from config.yaml
    """
    sensors: dict[str, SensorConfig] = field(default_factory=dict)
    migrated_from_config_yaml: bool = False
    
    def __post_init__(self):
        """Initialize sensors with defaults if not provided."""
        all_sensors = get_all_sensor_definitions()
        for sensor_def in all_sensors:
            if sensor_def.category_name not in self.sensors:
                # Default: enabled for core, disabled for experimental
                self.sensors[sensor_def.category_name] = SensorConfig(
                    category_name=sensor_def.category_name,
                    entity_id=sensor_def.default_entity_id,
                    enabled=sensor_def.is_core,
                )
    
    def get_sensor_config(self, category_name: str) -> Optional[SensorConfig]:
        """Get configuration for a specific sensor category."""
        return self.sensors.get(category_name)
    
    def set_entity_id(self, category_name: str, entity_id: str) -> bool:
        """
        Set the entity ID for a sensor category.
        
        Args:
            category_name: The sensor category name
            entity_id: The Home Assistant entity ID
            
        Returns:
            True if the sensor exists and was updated
        """
        if category_name in self.sensors:
            self.sensors[category_name].entity_id = entity_id
            return True
        return False
    
    def set_unit(self, category_name: str, unit: str) -> bool:
        """
        Set the unit for a sensor category.
        
        Args:
            category_name: The sensor category name
            unit: The unit of measurement (e.g., "°C", "kWh", "%")
            
        Returns:
            True if the sensor exists and was updated
        """
        if category_name in self.sensors:
            self.sensors[category_name].unit = unit
            return True
        return False
            return True
        return False
    
    def enable_sensor(self, category_name: str) -> bool:
        """
        Enable an experimental sensor.
        
        Args:
            category_name: The sensor category name
            
        Returns:
            True if the sensor was enabled (must be experimental)
        """
        sensor_def = get_sensor_definition(category_name)
        if sensor_def is None or sensor_def.is_core:
            return False
        
        if category_name in self.sensors:
            self.sensors[category_name].enabled = True
            return True
        return False
    
    def disable_sensor(self, category_name: str) -> bool:
        """
        Disable an experimental sensor.
        
        Args:
            category_name: The sensor category name
            
        Returns:
            True if the sensor was disabled (must be experimental)
        """
        sensor_def = get_sensor_definition(category_name)
        if sensor_def is None or sensor_def.is_core:
            return False
        
        if category_name in self.sensors:
            self.sensors[category_name].enabled = False
            return True
        return False
    
    def get_enabled_sensors(self) -> list[SensorConfig]:
        """Get all enabled sensor configurations."""
        return [s for s in self.sensors.values() if s.enabled]
    
    def get_enabled_sensor_entity_ids(self) -> list[str]:
        """Get entity IDs for all enabled sensors."""
        return [s.entity_id for s in self.get_enabled_sensors() if s.entity_id]
    
    def get_sensors_by_type(self) -> dict[str, list[dict]]:
        """
        Get sensors grouped by type for UI display.
        
        Returns:
            Dictionary with sensor type as key and list of sensor configs as value
        """
        all_defs = get_all_sensor_definitions()
        def_lookup = {d.category_name: d for d in all_defs}
        
        grouped: dict[str, list[dict]] = {}
        
        for category_name, config in self.sensors.items():
            sensor_def = def_lookup.get(category_name)
            if sensor_def is None:
                continue
            
            sensor_type = sensor_def.sensor_type.value
            if sensor_type not in grouped:
                grouped[sensor_type] = []
            
            grouped[sensor_type].append({
                "category_name": category_name,
                "display_name": sensor_def.display_name,
                "description": sensor_def.description,
                "unit": config.unit if config.unit else sensor_def.unit,  # Use config unit if set, otherwise default
                "is_core": sensor_def.is_core,
                "entity_id": config.entity_id,
                "enabled": config.enabled,
            })
        
        return grouped
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sensors": {k: v.to_dict() for k, v in self.sensors.items()},
            "migrated_from_config_yaml": self.migrated_from_config_yaml,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SensorCategoryConfiguration":
        """Create from dictionary."""
        sensors = {}
        if "sensors" in data:
            for category_name, sensor_data in data["sensors"].items():
                sensors[category_name] = SensorConfig.from_dict(sensor_data)
        
        config = cls(
            sensors=sensors,
            migrated_from_config_yaml=data.get("migrated_from_config_yaml", False),
        )
        
        return config
    
    def save(self) -> bool:
        """
        Save configuration to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            SENSOR_CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            with open(SENSOR_CONFIG_FILE_PATH, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            
            _Logger.info("Sensor category configuration saved to %s", SENSOR_CONFIG_FILE_PATH)
            return True
        except Exception as e:
            _Logger.error("Failed to save sensor category configuration: %s", e)
            return False
    
    @classmethod
    def load(cls) -> "SensorCategoryConfiguration":
        """
        Load configuration from disk.
        
        Returns:
            SensorCategoryConfiguration instance (default if file doesn't exist)
        """
        try:
            if SENSOR_CONFIG_FILE_PATH.exists():
                with open(SENSOR_CONFIG_FILE_PATH, "r") as f:
                    data = json.load(f)
                _Logger.info("Sensor category configuration loaded from %s", SENSOR_CONFIG_FILE_PATH)
                return cls.from_dict(data)
            else:
                _Logger.info("No sensor category configuration file, using defaults")
                return cls()
        except Exception as e:
            _Logger.error("Failed to load sensor category configuration: %s", e)
            return cls()


# Global configuration instance (singleton pattern)
_config: Optional[SensorCategoryConfiguration] = None


def get_sensor_category_config() -> SensorCategoryConfiguration:
    """
    Get the global sensor category configuration instance.
    
    Returns:
        SensorCategoryConfiguration singleton
    """
    global _config
    if _config is None:
        _config = SensorCategoryConfiguration.load()
        
        # Migrate from environment variables if not done yet
        if not _config.migrated_from_config_yaml:
            _migrate_from_env_vars(_config)
    
    return _config


def reload_sensor_category_config() -> SensorCategoryConfiguration:
    """
    Reload the sensor category configuration from disk.
    
    Returns:
        Reloaded SensorCategoryConfiguration
    """
    global _config
    _config = SensorCategoryConfiguration.load()
    return _config


def _migrate_from_env_vars(config: SensorCategoryConfiguration) -> None:
    """
    Migrate sensor configuration from environment variables.
    
    This is called once on first run to import settings from config.yaml
    which are exposed as environment variables.
    
    Args:
        config: The configuration instance to update
    """
    _Logger.info("Migrating sensor configuration from environment variables...")
    
    all_sensors = get_all_sensor_definitions()
    migrated_count = 0
    
    for sensor_def in all_sensors:
        entity_id = os.environ.get(sensor_def.env_var, "")
        if entity_id:
            if config.set_entity_id(sensor_def.category_name, entity_id):
                migrated_count += 1
                _Logger.debug(
                    "Migrated %s from %s: %s",
                    sensor_def.category_name,
                    sensor_def.env_var,
                    entity_id,
                )
            
            # Enable experimental sensors if they have a value from config.yaml
            if not sensor_def.is_core:
                config.enable_sensor(sensor_def.category_name)
    
    config.migrated_from_config_yaml = True
    config.save()
    
    _Logger.info("Migrated %d sensor configurations from environment variables", migrated_count)


def get_configured_sensor_entities() -> list[str]:
    """
    Get list of entity IDs for all enabled sensors.
    
    This function is used by the sensor sync worker to determine
    which entities to sync from Home Assistant.
    
    Returns:
        List of entity IDs for enabled sensors
    """
    config = get_sensor_category_config()
    return config.get_enabled_sensor_entity_ids()


def get_sensor_entity_id(category_name: str) -> Optional[str]:
    """
    Get the entity ID for a specific sensor category.
    
    Args:
        category_name: The sensor category name
        
    Returns:
        Entity ID or None if not configured or disabled
    """
    config = get_sensor_category_config()
    sensor_config = config.get_sensor_config(category_name)
    if sensor_config and sensor_config.enabled:
        return sensor_config.entity_id
    return None
