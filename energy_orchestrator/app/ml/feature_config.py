"""
Feature configuration for the heat pump consumption prediction model.

This module defines:
- The 13 core baseline features (always active, cannot be disabled)
- Experimental/optional features (disabled by default, toggleable via UI)
- Feature metadata (category, description, unit, time_window, is_core)
- Time zone configuration for time-based features

All features are derived from 5-minute averaged samples:
- 1 hour  = last 12 samples
- 6 hours = last 72 samples
- 24 hours = last 288 samples
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import json
import os
from zoneinfo import ZoneInfo

_Logger = logging.getLogger(__name__)

# Default timezone for time features (IANA identifier)
DEFAULT_TIMEZONE = "Europe/Amsterdam"

# Path to store feature configuration
CONFIG_DIR = os.environ.get("CONFIG_DIR", "/data")
CONFIG_FILENAME = "feature_config.json"


class FeatureCategory(str, Enum):
    """Categories for grouping features in the UI."""
    WEATHER = "weather"
    INDOOR = "indoor"
    CONTROL = "control"
    USAGE = "usage"
    TIME = "time"


class TimeWindow(str, Enum):
    """Time windows for aggregation features."""
    NONE = "none"
    HOUR_1 = "1h"
    HOUR_6 = "6h"
    HOUR_24 = "24h"
    DAY_7 = "7d"


@dataclass
class FeatureMetadata:
    """
    Metadata for a single feature.
    
    Attributes:
        name: Feature name used in the model
        category: Feature category for UI grouping
        description: Human-readable description
        unit: Unit of measurement (e.g., °C, kWh, %, m/s)
        time_window: Time window for aggregation features
        is_core: True for baseline features (always active)
        enabled: True if the feature is currently enabled
    """
    name: str
    category: FeatureCategory
    description: str
    unit: str
    time_window: TimeWindow = TimeWindow.NONE
    is_core: bool = True
    enabled: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "unit": self.unit,
            "time_window": self.time_window.value,
            "is_core": self.is_core,
            "enabled": self.enabled,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeatureMetadata":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            category=FeatureCategory(data["category"]),
            description=data["description"],
            unit=data["unit"],
            time_window=TimeWindow(data.get("time_window", "none")),
            is_core=data.get("is_core", True),
            enabled=data.get("enabled", True),
        )


# =============================================================================
# CORE BASELINE FEATURES (13 features - always active, cannot be disabled)
# =============================================================================

CORE_FEATURES: list[FeatureMetadata] = [
    # Weather / Outdoor (4 features)
    FeatureMetadata(
        name="outdoor_temp",
        category=FeatureCategory.WEATHER,
        description="Latest 5-minute outdoor temperature",
        unit="°C",
        time_window=TimeWindow.NONE,
        is_core=True,
    ),
    FeatureMetadata(
        name="outdoor_temp_avg_24h",
        category=FeatureCategory.WEATHER,
        description="24-hour outdoor temperature average (last 288 samples)",
        unit="°C",
        time_window=TimeWindow.HOUR_24,
        is_core=True,
    ),
    FeatureMetadata(
        name="wind",
        category=FeatureCategory.WEATHER,
        description="Latest 5-minute wind speed or intensity",
        unit="m/s",
        time_window=TimeWindow.NONE,
        is_core=True,
    ),
    FeatureMetadata(
        name="humidity",
        category=FeatureCategory.WEATHER,
        description="Latest 5-minute outdoor relative humidity",
        unit="%",
        time_window=TimeWindow.NONE,
        is_core=True,
    ),
    
    # Indoor Climate / Building Mass (2 features)
    FeatureMetadata(
        name="indoor_temp",
        category=FeatureCategory.INDOOR,
        description="Latest 5-minute indoor temperature",
        unit="°C",
        time_window=TimeWindow.NONE,
        is_core=True,
    ),
    FeatureMetadata(
        name="indoor_temp_avg_24h",
        category=FeatureCategory.INDOOR,
        description="24-hour average indoor temperature (last 288 samples), representing building mass / thermal history",
        unit="°C",
        time_window=TimeWindow.HOUR_24,
        is_core=True,
    ),
    
    # Control Behaviour / Setpoints (2 features)
    FeatureMetadata(
        name="target_temp",
        category=FeatureCategory.CONTROL,
        description="Latest 5-minute heating target setpoint",
        unit="°C",
        time_window=TimeWindow.NONE,
        is_core=True,
    ),
    FeatureMetadata(
        name="target_temp_avg_6h",
        category=FeatureCategory.CONTROL,
        description="6-hour average heating target setpoint (last 72 samples)",
        unit="°C",
        time_window=TimeWindow.HOUR_6,
        is_core=True,
    ),
    
    # Recent Consumption (3 features)
    FeatureMetadata(
        name="heating_kwh_last_1h",
        category=FeatureCategory.USAGE,
        description="Heating energy consumption in the last 1 hour (last 12 samples)",
        unit="kWh",
        time_window=TimeWindow.HOUR_1,
        is_core=True,
    ),
    FeatureMetadata(
        name="heating_kwh_last_6h",
        category=FeatureCategory.USAGE,
        description="Heating energy consumption in the last 6 hours (last 72 samples)",
        unit="kWh",
        time_window=TimeWindow.HOUR_6,
        is_core=True,
    ),
    FeatureMetadata(
        name="heating_kwh_last_24h",
        category=FeatureCategory.USAGE,
        description="Heating energy consumption in the last 24 hours (last 288 samples)",
        unit="kWh",
        time_window=TimeWindow.HOUR_24,
        is_core=True,
    ),
    
    # Time of Day (1 feature)
    FeatureMetadata(
        name="hour_of_day",
        category=FeatureCategory.TIME,
        description="Local hour of day (0-23), converted from UTC to configured timezone",
        unit="hour",
        time_window=TimeWindow.NONE,
        is_core=True,
    ),
    
    # Derived Domain Feature (1 feature)
    FeatureMetadata(
        name="delta_target_indoor",
        category=FeatureCategory.CONTROL,
        description="Difference between target and indoor temperature (target_temp - indoor_temp)",
        unit="°C",
        time_window=TimeWindow.NONE,
        is_core=True,
    ),
]


# =============================================================================
# EXPERIMENTAL FEATURES (optional, disabled by default, toggleable via UI)
# =============================================================================

EXPERIMENTAL_FEATURES: list[FeatureMetadata] = [
    # Weather - additional aggregations
    FeatureMetadata(
        name="pressure",
        category=FeatureCategory.WEATHER,
        description="Latest 5-minute barometric pressure",
        unit="hPa",
        time_window=TimeWindow.NONE,
        is_core=False,
        enabled=False,
    ),
    FeatureMetadata(
        name="outdoor_temp_avg_1h",
        category=FeatureCategory.WEATHER,
        description="1-hour outdoor temperature average (last 12 samples)",
        unit="°C",
        time_window=TimeWindow.HOUR_1,
        is_core=False,
        enabled=False,
    ),
    FeatureMetadata(
        name="outdoor_temp_avg_6h",
        category=FeatureCategory.WEATHER,
        description="6-hour outdoor temperature average (last 72 samples)",
        unit="°C",
        time_window=TimeWindow.HOUR_6,
        is_core=False,
        enabled=False,
    ),
    FeatureMetadata(
        name="outdoor_temp_avg_7d",
        category=FeatureCategory.WEATHER,
        description="7-day outdoor temperature average (last 2016 samples)",
        unit="°C",
        time_window=TimeWindow.DAY_7,
        is_core=False,
        enabled=False,
    ),
    
    # Indoor - additional aggregation
    FeatureMetadata(
        name="indoor_temp_avg_6h",
        category=FeatureCategory.INDOOR,
        description="6-hour average indoor temperature (last 72 samples)",
        unit="°C",
        time_window=TimeWindow.HOUR_6,
        is_core=False,
        enabled=False,
    ),
    
    # Control - additional aggregation
    FeatureMetadata(
        name="target_temp_avg_24h",
        category=FeatureCategory.CONTROL,
        description="24-hour average heating target setpoint (last 288 samples)",
        unit="°C",
        time_window=TimeWindow.HOUR_24,
        is_core=False,
        enabled=False,
    ),
    
    # Usage - 7-day window
    FeatureMetadata(
        name="heating_kwh_last_7d",
        category=FeatureCategory.USAGE,
        description="Heating energy consumption in the last 7 days (last 2016 samples)",
        unit="kWh",
        time_window=TimeWindow.DAY_7,
        is_core=False,
        enabled=False,
    ),
    FeatureMetadata(
        name="heating_degree_hours_24h",
        category=FeatureCategory.USAGE,
        description="Heating degree hours over 24h: sum of (target - outdoor)+ × (5/60)",
        unit="°C·h",
        time_window=TimeWindow.HOUR_24,
        is_core=False,
        enabled=False,
    ),
    FeatureMetadata(
        name="heating_degree_hours_7d",
        category=FeatureCategory.USAGE,
        description="Heating degree hours over 7 days",
        unit="°C·h",
        time_window=TimeWindow.DAY_7,
        is_core=False,
        enabled=False,
    ),
    
    # Time - additional calendar features
    FeatureMetadata(
        name="day_of_week",
        category=FeatureCategory.TIME,
        description="Day of week (0=Monday, 6=Sunday)",
        unit="day",
        time_window=TimeWindow.NONE,
        is_core=False,
        enabled=False,
    ),
    FeatureMetadata(
        name="is_weekend",
        category=FeatureCategory.TIME,
        description="1 if Saturday or Sunday, else 0",
        unit="boolean",
        time_window=TimeWindow.NONE,
        is_core=False,
        enabled=False,
    ),
    FeatureMetadata(
        name="is_night",
        category=FeatureCategory.TIME,
        description="1 if hour is 23:00-06:59, else 0",
        unit="boolean",
        time_window=TimeWindow.NONE,
        is_core=False,
        enabled=False,
    ),
]


@dataclass
class FeatureConfiguration:
    """
    Complete feature configuration for the model.
    
    This configuration is the single source of truth for:
    - Which features are used in training and prediction
    - Feature metadata for UI display and documentation
    - Timezone settings for time-based features
    - Two-step prediction settings (experimental)
    """
    timezone: str = DEFAULT_TIMEZONE
    experimental_enabled: dict[str, bool] = field(default_factory=dict)
    # Two-step prediction: first classify active/inactive, then regress for active hours only
    two_step_prediction_enabled: bool = False
    
    def __post_init__(self):
        """Initialize experimental feature states from defaults if not provided."""
        for feature in EXPERIMENTAL_FEATURES:
            if feature.name not in self.experimental_enabled:
                self.experimental_enabled[feature.name] = feature.enabled
    
    def get_all_features(self) -> list[FeatureMetadata]:
        """Get all features (core + experimental) with current enabled state."""
        features = []
        
        # Core features are always enabled
        for f in CORE_FEATURES:
            features.append(FeatureMetadata(
                name=f.name,
                category=f.category,
                description=f.description,
                unit=f.unit,
                time_window=f.time_window,
                is_core=True,
                enabled=True,
            ))
        
        # Experimental features use configured state
        for f in EXPERIMENTAL_FEATURES:
            features.append(FeatureMetadata(
                name=f.name,
                category=f.category,
                description=f.description,
                unit=f.unit,
                time_window=f.time_window,
                is_core=False,
                enabled=self.experimental_enabled.get(f.name, False),
            ))
        
        return features
    
    def get_active_features(self) -> list[FeatureMetadata]:
        """Get only active (enabled) features."""
        return [f for f in self.get_all_features() if f.enabled]
    
    def get_active_feature_names(self) -> list[str]:
        """Get names of active features only."""
        return [f.name for f in self.get_active_features()]
    
    def get_core_feature_names(self) -> list[str]:
        """Get names of core baseline features only."""
        return [f.name for f in CORE_FEATURES]
    
    def get_features_by_category(self) -> dict[str, list[FeatureMetadata]]:
        """Get features grouped by category for UI display."""
        features = self.get_all_features()
        grouped: dict[str, list[FeatureMetadata]] = {}
        
        for f in features:
            cat = f.category.value
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(f)
        
        return grouped
    
    def enable_experimental_feature(self, feature_name: str) -> bool:
        """
        Enable an experimental feature.
        
        Args:
            feature_name: Name of the feature to enable
            
        Returns:
            True if feature was found and enabled, False otherwise
        """
        for f in EXPERIMENTAL_FEATURES:
            if f.name == feature_name:
                self.experimental_enabled[feature_name] = True
                return True
        return False
    
    def disable_experimental_feature(self, feature_name: str) -> bool:
        """
        Disable an experimental feature.
        
        Args:
            feature_name: Name of the feature to disable
            
        Returns:
            True if feature was found and disabled, False otherwise
        """
        for f in EXPERIMENTAL_FEATURES:
            if f.name == feature_name:
                self.experimental_enabled[feature_name] = False
                return True
        return False
    
    def set_timezone(self, timezone: str) -> bool:
        """
        Set the timezone for time-based features.
        
        Args:
            timezone: IANA timezone identifier (e.g., 'Europe/Amsterdam')
            
        Returns:
            True if timezone is valid, False otherwise
        """
        try:
            # Validate timezone
            ZoneInfo(timezone)
            self.timezone = timezone
            return True
        except Exception as e:
            _Logger.warning("Invalid timezone '%s': %s", timezone, e)
            return False
    
    def get_timezone_info(self) -> ZoneInfo:
        """Get the configured timezone as a ZoneInfo object."""
        try:
            return ZoneInfo(self.timezone)
        except Exception:
            _Logger.warning("Invalid timezone '%s', falling back to %s", 
                          self.timezone, DEFAULT_TIMEZONE)
            return ZoneInfo(DEFAULT_TIMEZONE)
    
    def enable_two_step_prediction(self) -> None:
        """Enable the two-step prediction mode (classifier + regressor)."""
        self.two_step_prediction_enabled = True
        _Logger.info("Two-step prediction enabled")
    
    def disable_two_step_prediction(self) -> None:
        """Disable the two-step prediction mode (use single regressor)."""
        self.two_step_prediction_enabled = False
        _Logger.info("Two-step prediction disabled")
    
    def is_two_step_prediction_enabled(self) -> bool:
        """Check if two-step prediction is enabled."""
        return self.two_step_prediction_enabled
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timezone": self.timezone,
            "experimental_enabled": self.experimental_enabled,
            "two_step_prediction_enabled": self.two_step_prediction_enabled,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeatureConfiguration":
        """Create from dictionary."""
        return cls(
            timezone=data.get("timezone", DEFAULT_TIMEZONE),
            experimental_enabled=data.get("experimental_enabled", {}),
            two_step_prediction_enabled=data.get("two_step_prediction_enabled", False),
        )
    
    def save(self) -> bool:
        """
        Save configuration to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            config_path = os.path.join(CONFIG_DIR, CONFIG_FILENAME)
            os.makedirs(CONFIG_DIR, exist_ok=True)
            
            with open(config_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            
            _Logger.info("Feature configuration saved to %s", config_path)
            return True
        except Exception as e:
            _Logger.error("Failed to save feature configuration: %s", e)
            return False
    
    @classmethod
    def load(cls) -> "FeatureConfiguration":
        """
        Load configuration from disk.
        
        Returns:
            FeatureConfiguration instance (default if file doesn't exist)
        """
        try:
            config_path = os.path.join(CONFIG_DIR, CONFIG_FILENAME)
            
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    data = json.load(f)
                _Logger.info("Feature configuration loaded from %s", config_path)
                return cls.from_dict(data)
            else:
                _Logger.info("No feature configuration file, using defaults")
                return cls()
        except Exception as e:
            _Logger.error("Failed to load feature configuration: %s", e)
            return cls()


# Global configuration instance (singleton pattern)
_config: Optional[FeatureConfiguration] = None


def get_feature_config() -> FeatureConfiguration:
    """
    Get the global feature configuration instance.
    
    Returns:
        FeatureConfiguration singleton
    """
    global _config
    if _config is None:
        _config = FeatureConfiguration.load()
    return _config


def reload_feature_config() -> FeatureConfiguration:
    """
    Reload the feature configuration from disk.
    
    Returns:
        Reloaded FeatureConfiguration
    """
    global _config
    _config = FeatureConfiguration.load()
    return _config


def convert_utc_to_local_hour(utc_timestamp: datetime, config: Optional[FeatureConfiguration] = None) -> int:
    """
    Convert a UTC timestamp to local hour using configured timezone.
    
    Args:
        utc_timestamp: Timestamp in UTC (naive or aware)
        config: Optional feature configuration (uses global if not provided)
        
    Returns:
        Local hour of day (0-23)
    """
    if config is None:
        config = get_feature_config()
    
    try:
        tz = config.get_timezone_info()
        
        # Make timestamp timezone-aware if naive
        if utc_timestamp.tzinfo is None:
            utc_timestamp = utc_timestamp.replace(tzinfo=timezone.utc)
        
        # Convert to local timezone
        local_time = utc_timestamp.astimezone(tz)
        return local_time.hour
    except Exception as e:
        _Logger.warning("Error converting timezone: %s, using UTC hour", e)
        return utc_timestamp.hour


def get_feature_metadata_dict() -> list[dict]:
    """
    Get all feature metadata as a list of dictionaries.
    
    This is useful for API responses and documentation.
    
    Returns:
        List of feature metadata dictionaries
    """
    config = get_feature_config()
    return [f.to_dict() for f in config.get_all_features()]


def get_core_feature_count() -> int:
    """Get the count of core baseline features."""
    return len(CORE_FEATURES)


def validate_feature_set(feature_names: list[str]) -> tuple[bool, list[str]]:
    """
    Validate that a list of feature names contains all core features.
    
    Args:
        feature_names: List of feature names to validate
        
    Returns:
        Tuple of (is_valid, missing_core_features)
    """
    core_names = {f.name for f in CORE_FEATURES}
    provided = set(feature_names)
    missing = core_names - provided
    
    return len(missing) == 0, list(missing)


# Raw sensor features (directly from sensors, no computation)
# These map directly to sensor values without any aggregation or derivation
RAW_SENSOR_FEATURES = {
    "outdoor_temp",
    "wind",
    "humidity",
    "pressure",
    "indoor_temp",
    "target_temp",
}

# Calculated/derived features (computed from raw data)
# These are computed through aggregation, derivation, or time-based logic
CALCULATED_FEATURES = {
    # Aggregated averages
    "outdoor_temp_avg_1h",
    "outdoor_temp_avg_6h",
    "outdoor_temp_avg_24h",
    "outdoor_temp_avg_7d",
    "indoor_temp_avg_6h",
    "indoor_temp_avg_24h",
    "target_temp_avg_6h",
    "target_temp_avg_24h",
    # Historical usage (computed from kWh sensor deltas)
    "heating_kwh_last_1h",
    "heating_kwh_last_6h",
    "heating_kwh_last_24h",
    "heating_kwh_last_7d",
    # Heating degree hours (computed from temperature differences)
    "heating_degree_hours_24h",
    "heating_degree_hours_7d",
    # Derived from multiple inputs
    "delta_target_indoor",
    # Time-based features
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
}


def categorize_features(feature_names: list[str]) -> dict[str, list[str]]:
    """
    Categorize features into raw sensor features and calculated features.
    
    Args:
        feature_names: List of feature names to categorize
        
    Returns:
        Dictionary with 'raw_sensor_features' and 'calculated_features' lists
    """
    raw = []
    calculated = []
    unknown = []
    
    for name in feature_names:
        if name in RAW_SENSOR_FEATURES:
            raw.append(name)
        elif name in CALCULATED_FEATURES:
            calculated.append(name)
        else:
            # Unknown features logged and categorized as calculated
            _Logger.debug("Unknown feature '%s' categorized as calculated", name)
            unknown.append(name)
            calculated.append(name)
    
    result = {
        "raw_sensor_features": raw,
        "calculated_features": calculated,
    }
    
    # Include unknown features in response if any were found
    if unknown:
        result["unknown_features"] = unknown
    
    return result


def get_feature_details(feature_names: list[str]) -> list[dict]:
    """
    Get detailed information about features including metadata.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        List of feature detail dictionaries
    """
    all_features = CORE_FEATURES + EXPERIMENTAL_FEATURES
    feature_lookup = {f.name: f for f in all_features}
    
    details = []
    for name in feature_names:
        if name in feature_lookup:
            f = feature_lookup[name]
            details.append({
                "name": name,
                "category": f.category.value,
                "description": f.description,
                "unit": f.unit,
                "time_window": f.time_window.value,
                "is_core": f.is_core,
                "is_calculated": name in CALCULATED_FEATURES,
            })
        else:
            # Unknown feature
            details.append({
                "name": name,
                "category": "unknown",
                "description": f"Feature '{name}'",
                "unit": "",
                "time_window": "none",
                "is_core": False,
                "is_calculated": name in CALCULATED_FEATURES,
            })
    
    return details


def verify_model_features(
    model_feature_names: list[str],
    dataset_feature_names: list[str],
) -> dict:
    """
    Verify that model features match the dataset features.
    
    This function confirms that the features the model was trained with
    are actually present in the training dataset.
    
    Args:
        model_feature_names: Features the model expects
        dataset_feature_names: Features available in the dataset
        
    Returns:
        Dictionary with verification results
    """
    model_set = set(model_feature_names)
    dataset_set = set(dataset_feature_names)
    
    # Features in model that are in dataset (verified)
    verified = model_set & dataset_set
    
    # Features in model but not in dataset (missing from data)
    missing_in_dataset = model_set - dataset_set
    
    # Features in dataset but not used by model
    unused_in_model = dataset_set - model_set
    
    return {
        "verified": len(missing_in_dataset) == 0,
        "feature_count": len(model_feature_names),
        "verified_features": sorted(list(verified)),
        "missing_in_dataset": sorted(list(missing_in_dataset)),
        "unused_in_model": sorted(list(unused_in_model)),
    }
