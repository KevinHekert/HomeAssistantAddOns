"""
Virtual sensor configuration and management.

This module allows creating derived sensors based on two existing sensors
using mathematical operations like delta (subtraction), sum, multiply, divide, etc.

Example virtual sensors:
- temp_delta = target_temp - indoor_temp (heating demand indicator)
- total_power = device1_power + device2_power
- efficiency = output_energy / input_energy

Virtual sensors are:
- Calculated during resampling from raw sensor values
- Stored as categories in the resampled_samples table
- Available for feature generation (avg_1h, avg_6h, etc.)
- Usable in model training like any other sensor
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

_Logger = logging.getLogger(__name__)

# Configuration file path for virtual sensor definitions
_data_dir_env = os.environ.get("DATA_DIR", "/data")
_resolved_data_dir = Path(_data_dir_env).resolve()
_allowed_prefixes = ("/data", "/tmp", str(Path.cwd()))
if not any(_resolved_data_dir.as_posix().startswith(prefix) for prefix in _allowed_prefixes):
    _Logger.warning(
        "DATA_DIR '%s' is not in an allowed location, using default /data",
        _data_dir_env,
    )
    _resolved_data_dir = Path("/data")
DATA_DIR = _resolved_data_dir
VIRTUAL_SENSORS_CONFIG_FILE = DATA_DIR / "virtual_sensors_config.json"


class VirtualSensorOperation(str, Enum):
    """Mathematical operations supported for virtual sensors."""
    SUBTRACT = "subtract"  # sensor1 - sensor2
    ADD = "add"            # sensor1 + sensor2
    MULTIPLY = "multiply"  # sensor1 * sensor2
    DIVIDE = "divide"      # sensor1 / sensor2 (with zero check)
    AVERAGE = "average"    # (sensor1 + sensor2) / 2


@dataclass
class VirtualSensorDefinition:
    """
    Definition of a virtual (derived) sensor.
    
    Attributes:
        name: Unique name for the virtual sensor (e.g., "temp_delta")
        display_name: Human-readable name for UI display
        description: Description of what this sensor represents
        source_sensor1: Category name of first source sensor
        source_sensor2: Category name of second source sensor
        operation: Mathematical operation to apply
        unit: Unit of measurement for the result
        enabled: Whether this virtual sensor is currently enabled
    """
    name: str
    display_name: str
    description: str
    source_sensor1: str
    source_sensor2: str
    operation: VirtualSensorOperation
    unit: str = ""
    enabled: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "source_sensor1": self.source_sensor1,
            "source_sensor2": self.source_sensor2,
            "operation": self.operation.value,
            "unit": self.unit,
            "enabled": self.enabled,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VirtualSensorDefinition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            description=data["description"],
            source_sensor1=data["source_sensor1"],
            source_sensor2=data["source_sensor2"],
            operation=VirtualSensorOperation(data["operation"]),
            unit=data.get("unit", ""),
            enabled=data.get("enabled", True),
        )
    
    def calculate(self, value1: Optional[float], value2: Optional[float]) -> Optional[float]:
        """
        Calculate the virtual sensor value from two source values.
        
        Args:
            value1: Value from source_sensor1
            value2: Value from source_sensor2
            
        Returns:
            Calculated value, or None if inputs are invalid
        """
        if value1 is None or value2 is None:
            return None
        
        try:
            if self.operation == VirtualSensorOperation.SUBTRACT:
                return value1 - value2
            elif self.operation == VirtualSensorOperation.ADD:
                return value1 + value2
            elif self.operation == VirtualSensorOperation.MULTIPLY:
                return value1 * value2
            elif self.operation == VirtualSensorOperation.DIVIDE:
                if value2 == 0:
                    _Logger.warning("Division by zero in virtual sensor '%s'", self.name)
                    return None
                return value1 / value2
            elif self.operation == VirtualSensorOperation.AVERAGE:
                return (value1 + value2) / 2.0
            else:
                _Logger.error("Unknown operation '%s' for virtual sensor '%s'", 
                            self.operation, self.name)
                return None
        except Exception as e:
            _Logger.error("Error calculating virtual sensor '%s': %s", self.name, e)
            return None


@dataclass
class VirtualSensorsConfiguration:
    """
    Complete virtual sensors configuration.
    
    Stores all virtual sensor definitions.
    """
    sensors: dict[str, VirtualSensorDefinition] = field(default_factory=dict)
    
    def add_sensor(self, sensor: VirtualSensorDefinition) -> bool:
        """
        Add a new virtual sensor definition.
        
        Args:
            sensor: Virtual sensor definition to add
            
        Returns:
            True if added successfully, False if name already exists
        """
        if sensor.name in self.sensors:
            _Logger.warning("Virtual sensor '%s' already exists", sensor.name)
            return False
        
        self.sensors[sensor.name] = sensor
        return True
    
    def remove_sensor(self, name: str) -> bool:
        """
        Remove a virtual sensor definition.
        
        Args:
            name: Name of the virtual sensor to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self.sensors:
            del self.sensors[name]
            return True
        return False
    
    def get_sensor(self, name: str) -> Optional[VirtualSensorDefinition]:
        """Get a virtual sensor definition by name."""
        return self.sensors.get(name)
    
    def enable_sensor(self, name: str) -> bool:
        """Enable a virtual sensor."""
        if name in self.sensors:
            self.sensors[name].enabled = True
            return True
        return False
    
    def disable_sensor(self, name: str) -> bool:
        """Disable a virtual sensor."""
        if name in self.sensors:
            self.sensors[name].enabled = False
            return True
        return False
    
    def get_enabled_sensors(self) -> list[VirtualSensorDefinition]:
        """Get all enabled virtual sensors."""
        return [s for s in self.sensors.values() if s.enabled]
    
    def get_all_sensors(self) -> list[VirtualSensorDefinition]:
        """Get all virtual sensors (enabled and disabled)."""
        return list(self.sensors.values())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sensors": {k: v.to_dict() for k, v in self.sensors.items()},
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VirtualSensorsConfiguration":
        """Create from dictionary."""
        sensors = {}
        if "sensors" in data:
            for name, sensor_data in data["sensors"].items():
                sensors[name] = VirtualSensorDefinition.from_dict(sensor_data)
        
        return cls(sensors=sensors)
    
    def save(self) -> bool:
        """
        Save configuration to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            VIRTUAL_SENSORS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            with open(VIRTUAL_SENSORS_CONFIG_FILE, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            
            _Logger.info("Virtual sensors configuration saved to %s", VIRTUAL_SENSORS_CONFIG_FILE)
            return True
        except Exception as e:
            _Logger.error("Failed to save virtual sensors configuration: %s", e)
            return False
    
    @classmethod
    def load(cls) -> "VirtualSensorsConfiguration":
        """
        Load configuration from disk.
        
        Returns:
            VirtualSensorsConfiguration instance (default if file doesn't exist)
        """
        try:
            if VIRTUAL_SENSORS_CONFIG_FILE.exists():
                with open(VIRTUAL_SENSORS_CONFIG_FILE, "r") as f:
                    data = json.load(f)
                _Logger.info("Virtual sensors configuration loaded from %s", VIRTUAL_SENSORS_CONFIG_FILE)
                return cls.from_dict(data)
            else:
                _Logger.info("No virtual sensors configuration file, using defaults")
                return cls()
        except Exception as e:
            _Logger.error("Failed to load virtual sensors configuration: %s", e)
            return cls()


# Global configuration instance (singleton pattern)
_config: Optional[VirtualSensorsConfiguration] = None


def get_virtual_sensors_config() -> VirtualSensorsConfiguration:
    """
    Get the global virtual sensors configuration instance.
    
    Returns:
        VirtualSensorsConfiguration singleton
    """
    global _config
    if _config is None:
        _config = VirtualSensorsConfiguration.load()
    return _config


def reload_virtual_sensors_config() -> VirtualSensorsConfiguration:
    """
    Reload the virtual sensors configuration from disk.
    
    Returns:
        Reloaded VirtualSensorsConfiguration
    """
    global _config
    _config = VirtualSensorsConfiguration.load()
    return _config


def reset_virtual_sensors_config() -> None:
    """
    Reset the global virtual sensors configuration to None.
    
    This function is primarily for testing purposes, allowing tests to
    reset the singleton state between test runs.
    """
    global _config
    _config = None
