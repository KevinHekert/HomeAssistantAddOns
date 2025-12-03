"""
Feature statistics configuration for sensors.

This module manages which time-based aggregation statistics should be
generated for each sensor during resampling.

Supported statistics:
- avg_1h: 1-hour average
- avg_6h: 6-hour average
- avg_24h: 24-hour average
- avg_7d: 7-day average

These statistics can be enabled/disabled per sensor and will be calculated
during the resampling process and stored in the resampled_samples table
with category names like "sensor_name_avg_1h".
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

_Logger = logging.getLogger(__name__)

# Configuration file path
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
FEATURE_STATS_CONFIG_FILE = DATA_DIR / "feature_stats_config.json"


class StatType(str, Enum):
    """Time-based statistics that can be generated."""
    AVG_1H = "avg_1h"
    AVG_6H = "avg_6h"
    AVG_24H = "avg_24h"
    AVG_7D = "avg_7d"


# Default stats to enable for different sensor types
DEFAULT_ENABLED_STATS = {
    StatType.AVG_1H,
    StatType.AVG_6H,
    StatType.AVG_24H,
}


@dataclass
class SensorStatsConfig:
    """
    Configuration for which statistics to generate for a specific sensor.
    
    Attributes:
        sensor_name: Name/category of the sensor
        enabled_stats: Set of StatType values indicating which stats to generate
    """
    sensor_name: str
    enabled_stats: set[StatType] = field(default_factory=lambda: set(DEFAULT_ENABLED_STATS))
    
    def enable_stat(self, stat_type: StatType) -> None:
        """Enable a specific statistic type."""
        self.enabled_stats.add(stat_type)
    
    def disable_stat(self, stat_type: StatType) -> None:
        """Disable a specific statistic type."""
        self.enabled_stats.discard(stat_type)
    
    def is_stat_enabled(self, stat_type: StatType) -> bool:
        """Check if a specific statistic type is enabled."""
        return stat_type in self.enabled_stats
    
    def get_stat_category_name(self, stat_type: StatType) -> str:
        """Get the category name for a statistic (e.g., 'outdoor_temp_avg_1h')."""
        return f"{self.sensor_name}_{stat_type.value}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sensor_name": self.sensor_name,
            "enabled_stats": [s.value for s in self.enabled_stats],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SensorStatsConfig":
        """Create from dictionary."""
        # Only apply defaults if 'enabled_stats' key is completely missing
        # If the key exists (even as empty list), respect that explicit choice
        if "enabled_stats" in data:
            enabled_stats = {StatType(s) for s in data["enabled_stats"]}
        else:
            # Key missing - apply defaults for backward compatibility
            enabled_stats = set(DEFAULT_ENABLED_STATS)
        
        return cls(
            sensor_name=data["sensor_name"],
            enabled_stats=enabled_stats,
        )


@dataclass
class FeatureStatsConfiguration:
    """
    Complete feature statistics configuration.
    
    Manages which time-based statistics should be generated for each sensor.
    """
    sensor_configs: dict[str, SensorStatsConfig] = field(default_factory=dict)
    
    def get_sensor_config(self, sensor_name: str) -> SensorStatsConfig:
        """
        Get or create configuration for a sensor.
        
        Args:
            sensor_name: Name/category of the sensor
            
        Returns:
            SensorStatsConfig for the sensor (creates with defaults if not exists)
        """
        if sensor_name not in self.sensor_configs:
            self.sensor_configs[sensor_name] = SensorStatsConfig(sensor_name=sensor_name)
        return self.sensor_configs[sensor_name]
    
    def set_stat_enabled(self, sensor_name: str, stat_type: StatType, enabled: bool) -> None:
        """
        Enable or disable a statistic for a sensor.
        
        Args:
            sensor_name: Name/category of the sensor
            stat_type: Type of statistic
            enabled: Whether to enable or disable
        """
        config = self.get_sensor_config(sensor_name)
        if enabled:
            config.enable_stat(stat_type)
        else:
            config.disable_stat(stat_type)
    
    def get_enabled_stats_for_sensor(self, sensor_name: str) -> list[StatType]:
        """Get list of enabled statistics for a sensor."""
        config = self.get_sensor_config(sensor_name)
        return sorted(list(config.enabled_stats), key=lambda s: s.value)
    
    def get_all_enabled_stat_categories(self) -> list[str]:
        """
        Get list of all enabled statistic category names across all sensors.
        
        Returns category names like 'outdoor_temp_avg_1h', 'wind_avg_6h', etc.
        """
        categories = []
        for config in self.sensor_configs.values():
            for stat_type in config.enabled_stats:
                categories.append(config.get_stat_category_name(stat_type))
        return sorted(categories)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sensor_configs": {k: v.to_dict() for k, v in self.sensor_configs.items()},
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeatureStatsConfiguration":
        """Create from dictionary."""
        sensor_configs = {}
        if "sensor_configs" in data:
            for sensor_name, config_data in data["sensor_configs"].items():
                sensor_configs[sensor_name] = SensorStatsConfig.from_dict(config_data)
        
        return cls(sensor_configs=sensor_configs)
    
    def save(self) -> bool:
        """
        Save configuration to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            FEATURE_STATS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            with open(FEATURE_STATS_CONFIG_FILE, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            
            _Logger.info("Feature stats configuration saved to %s", FEATURE_STATS_CONFIG_FILE)
            return True
        except Exception as e:
            _Logger.error("Failed to save feature stats configuration: %s", e)
            return False
    
    @classmethod
    def load(cls) -> "FeatureStatsConfiguration":
        """
        Load configuration from disk.
        
        Returns:
            FeatureStatsConfiguration instance (default if file doesn't exist)
        """
        try:
            if FEATURE_STATS_CONFIG_FILE.exists():
                with open(FEATURE_STATS_CONFIG_FILE, "r") as f:
                    data = json.load(f)
                _Logger.info("Feature stats configuration loaded from %s", FEATURE_STATS_CONFIG_FILE)
                return cls.from_dict(data)
            else:
                _Logger.info("No feature stats configuration file, using defaults")
                return cls()
        except Exception as e:
            _Logger.error("Failed to load feature stats configuration: %s", e)
            return cls()


# Global configuration instance (singleton pattern)
_config: Optional[FeatureStatsConfiguration] = None


def get_feature_stats_config() -> FeatureStatsConfiguration:
    """
    Get the global feature statistics configuration instance.
    
    Returns:
        FeatureStatsConfiguration singleton
    """
    global _config
    if _config is None:
        _config = FeatureStatsConfiguration.load()
    return _config


def reload_feature_stats_config() -> FeatureStatsConfiguration:
    """
    Reload the feature statistics configuration from disk.
    
    Returns:
        Reloaded FeatureStatsConfiguration
    """
    global _config
    _config = FeatureStatsConfiguration.load()
    return _config


def derive_stats_from_feature_config() -> dict[str, set[StatType]]:
    """
    Determine which statistics should be enabled based on ML feature configuration.
    
    This function analyzes the active features in feature_config and determines
    which sensor statistics (avg_1h, avg_6h, etc.) need to be calculated.
    
    For example:
    - If outdoor_temp_avg_1h is active, then outdoor_temp needs AVG_1H enabled
    - If heating_kwh_last_24h is active, this is calculated in heating_features.py,
      not in feature statistics
    
    Returns:
        Dictionary mapping sensor names to sets of required StatTypes
        
    Example:
        {
            "outdoor_temp": {StatType.AVG_1H, StatType.AVG_24H},
            "indoor_temp": {StatType.AVG_6H, StatType.AVG_24H},
            ...
        }
    """
    from ml.feature_config import get_feature_config
    
    config = get_feature_config()
    active_features = config.get_active_feature_names()
    
    # Map feature names to (sensor_name, stat_type) pairs
    # Features like "outdoor_temp_avg_1h" -> ("outdoor_temp", StatType.AVG_1H)
    sensor_stats: dict[str, set[StatType]] = {}
    
    for feature_name in active_features:
        # Check if this is an aggregation feature (contains avg_)
        if "_avg_" in feature_name:
            parts = feature_name.rsplit("_avg_", 1)
            if len(parts) == 2:
                sensor_name = parts[0]
                time_window = parts[1]
                
                # Map time window to StatType
                stat_type = None
                if time_window == "1h":
                    stat_type = StatType.AVG_1H
                elif time_window == "6h":
                    stat_type = StatType.AVG_6H
                elif time_window == "24h":
                    stat_type = StatType.AVG_24H
                elif time_window == "7d":
                    stat_type = StatType.AVG_7D
                
                if stat_type:
                    if sensor_name not in sensor_stats:
                        sensor_stats[sensor_name] = set()
                    sensor_stats[sensor_name].add(stat_type)
    
    return sensor_stats


def sync_stats_config_with_features() -> FeatureStatsConfiguration:
    """
    Synchronize feature statistics configuration with ML feature configuration.
    
    This ensures that only the statistics needed by active ML features are enabled.
    
    Returns:
        Updated FeatureStatsConfiguration
    """
    stats_config = get_feature_stats_config()
    required_stats = derive_stats_from_feature_config()
    
    _Logger.info(
        "Syncing feature stats configuration with ML feature config: %d sensors require stats",
        len(required_stats),
    )
    
    # Update each sensor's configuration
    for sensor_name, stat_types in required_stats.items():
        sensor_config = stats_config.get_sensor_config(sensor_name)
        sensor_config.enabled_stats = stat_types
    
    # Disable stats for sensors not in required_stats
    for sensor_name in list(stats_config.sensor_configs.keys()):
        if sensor_name not in required_stats:
            sensor_config = stats_config.get_sensor_config(sensor_name)
            sensor_config.enabled_stats = set()
    
    # Save updated configuration
    stats_config.save()
    
    return stats_config
