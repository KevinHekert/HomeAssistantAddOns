"""
Weather API integration for weerlive.nl.

This module provides functionality to:
- Fetch weather forecast data from weerlive.nl API
- Store and manage API settings (API key, location)
- Parse weather forecast for use in scenario predictions

API Documentation: https://weerlive.nl/delen.php
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib import request, error

_Logger = logging.getLogger(__name__)

# Configuration file path for persistent weather settings storage
# In Home Assistant add-ons, /data is the persistent data directory
WEATHER_CONFIG_FILE_PATH = Path(os.environ.get("DATA_DIR", "/data")) / "weather_config.json"

# API base URL for weerlive.nl
WEERLIVE_API_URL = "https://weerlive.nl/api/weerlive_api_v2.php"

# Cache for weather configuration to avoid repeated file reads
_cached_config: Optional["WeatherConfig"] = None
_cached_mtime: Optional[float] = None


@dataclass
class WeatherConfig:
    """Configuration for weather API access."""

    api_key: str = ""
    location: str = ""


@dataclass
class HourlyForecast:
    """Weather forecast for a specific hour."""

    timestamp: datetime
    temperature: float  # Temperature in °C
    wind_speed: float  # Wind speed in m/s
    humidity: float  # Humidity in %
    pressure: float  # Pressure in hPa
    precipitation: float = 0.0  # Precipitation in mm
    description: str = ""  # Weather description


@dataclass
class WeatherForecastResult:
    """Result from weather forecast API call."""

    success: bool
    hourly_forecasts: list[HourlyForecast] = field(default_factory=list)
    error_message: str | None = None
    location_name: str = ""
    current_temp: float | None = None


def _invalidate_cache() -> None:
    """Invalidate the cached configuration."""
    global _cached_config, _cached_mtime
    _cached_config = None
    _cached_mtime = None


def _get_file_mtime() -> Optional[float]:
    """Get the modification time of the config file, or None if it doesn't exist."""
    try:
        if WEATHER_CONFIG_FILE_PATH.exists():
            return WEATHER_CONFIG_FILE_PATH.stat().st_mtime
    except OSError:
        pass
    return None


def _load_weather_config() -> WeatherConfig:
    """Load weather configuration from persistent file with caching.

    Returns:
        WeatherConfig with values from file or defaults if not configured.
    """
    global _cached_config, _cached_mtime

    # Check if we can use cached config
    current_mtime = _get_file_mtime()
    if _cached_config is not None and current_mtime == _cached_mtime:
        return _cached_config

    config = WeatherConfig()

    try:
        if WEATHER_CONFIG_FILE_PATH.exists():
            with open(WEATHER_CONFIG_FILE_PATH, "r") as f:
                data = json.load(f)

            if "api_key" in data and isinstance(data["api_key"], str):
                config.api_key = data["api_key"]

            if "location" in data and isinstance(data["location"], str):
                config.location = data["location"]

    except (json.JSONDecodeError, OSError) as e:
        _Logger.warning("Error loading weather config: %s. Using defaults.", e)

    # Update cache
    _cached_config = config
    _cached_mtime = current_mtime

    return config


def _save_weather_config(config: WeatherConfig) -> bool:
    """Save weather configuration to persistent file.

    Args:
        config: WeatherConfig to save.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        # Ensure parent directory exists
        WEATHER_CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "api_key": config.api_key,
            "location": config.location,
        }

        with open(WEATHER_CONFIG_FILE_PATH, "w") as f:
            json.dump(data, f, indent=2)

        # Invalidate cache so next read picks up new values
        _invalidate_cache()

        _Logger.info("Weather config saved for location: %s", config.location)
        return True

    except OSError as e:
        _Logger.error("Error saving weather config: %s", e)
        return False


def get_weather_config() -> WeatherConfig:
    """Get the current weather configuration.

    Returns:
        WeatherConfig with current values.
    """
    return _load_weather_config()


def set_weather_config(api_key: str | None = None, location: str | None = None) -> tuple[bool, str | None]:
    """Update weather configuration values.

    Only provided (non-None) values are updated.

    Args:
        api_key: Weerlive.nl API key.
        location: Location for weather forecast (city name or coordinates).

    Returns:
        Tuple of (success, error_message).
        error_message is None if successful.
    """
    config = _load_weather_config()

    if api_key is not None:
        if not isinstance(api_key, str):
            return False, "api_key must be a string"
        config.api_key = api_key.strip()

    if location is not None:
        if not isinstance(location, str):
            return False, "location must be a string"
        config.location = location.strip()

    if _save_weather_config(config):
        return True, None
    return False, "Failed to save configuration"


def validate_weather_api(api_key: str, location: str) -> tuple[bool, str | None, str | None]:
    """Validate weather API credentials by making a test request.

    Args:
        api_key: Weerlive.nl API key.
        location: Location for weather forecast.

    Returns:
        Tuple of (success, error_message, location_name).
        If successful, returns the resolved location name.
    """
    if not api_key:
        return False, "API key is required", None
    if not location:
        return False, "Location is required", None

    url = f"{WEERLIVE_API_URL}?key={api_key}&locatie={location}"

    try:
        req = request.Request(url)
        req.add_header("User-Agent", "HomeAssistant-EnergyOrchestrator/1.0")

        with request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Check for API errors
        if "liveweer" not in data or len(data["liveweer"]) == 0:
            return False, "Invalid API response - no weather data returned", None

        live_data = data["liveweer"][0]

        # Check for explicit error in response
        if "fpiurl" in live_data and not live_data.get("temp"):
            return False, "API returned error - check API key and location", None

        location_name = live_data.get("plaats", location)
        return True, None, location_name

    except error.HTTPError as e:
        if e.code == 401:
            return False, "Invalid API key", None
        return False, f"HTTP error: {e.code}", None
    except error.URLError as e:
        return False, f"Connection error: {e.reason}", None
    except json.JSONDecodeError:
        return False, "Invalid response from API", None
    except Exception as e:
        _Logger.error("Unexpected error validating weather API: %s", e)
        return False, f"Unexpected error: {str(e)}", None


def fetch_weather_forecast(api_key: str | None = None, location: str | None = None) -> WeatherForecastResult:
    """Fetch weather forecast from weerlive.nl API.

    If api_key and location are not provided, uses stored configuration.

    The weerlive.nl API returns hourly forecast data for the next 24 hours.

    Args:
        api_key: Optional API key. Uses stored config if not provided.
        location: Optional location. Uses stored config if not provided.

    Returns:
        WeatherForecastResult with hourly forecasts or error information.
    """
    # Use stored config if not provided
    config = get_weather_config()
    api_key = api_key or config.api_key
    location = location or config.location

    if not api_key:
        return WeatherForecastResult(
            success=False,
            error_message="API key not configured. Please set up weather API in settings.",
        )
    if not location:
        return WeatherForecastResult(
            success=False,
            error_message="Location not configured. Please set up weather API in settings.",
        )

    url = f"{WEERLIVE_API_URL}?key={api_key}&locatie={location}"

    try:
        req = request.Request(url)
        req.add_header("User-Agent", "HomeAssistant-EnergyOrchestrator/1.0")

        _Logger.debug("Fetching weather forecast from: %s", url)
        with request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Parse response
        if "liveweer" not in data or len(data["liveweer"]) == 0:
            return WeatherForecastResult(
                success=False,
                error_message="Invalid API response - no weather data returned",
            )

        live_data = data["liveweer"][0]
        location_name = live_data.get("plaats", location)

        # Get current temperature
        current_temp = None
        try:
            current_temp = float(live_data.get("temp", 0))
        except (TypeError, ValueError):
            pass

        # Parse hourly forecasts
        # The Weerlive API v2 provides hourly forecasts in 'uur_verw' at root level
        # Also check inside live_data for backwards compatibility
        hourly_forecasts = _parse_hourly_forecasts(data, live_data)

        if not hourly_forecasts:
            return WeatherForecastResult(
                success=False,
                error_message="No hourly forecast data available",
                location_name=location_name,
                current_temp=current_temp,
            )

        return WeatherForecastResult(
            success=True,
            hourly_forecasts=hourly_forecasts,
            location_name=location_name,
            current_temp=current_temp,
        )

    except error.HTTPError as e:
        return WeatherForecastResult(
            success=False,
            error_message=f"HTTP error: {e.code}",
        )
    except error.URLError as e:
        return WeatherForecastResult(
            success=False,
            error_message=f"Connection error: {e.reason}",
        )
    except json.JSONDecodeError:
        return WeatherForecastResult(
            success=False,
            error_message="Invalid response from API",
        )
    except Exception as e:
        _Logger.error("Unexpected error fetching weather forecast: %s", e)
        return WeatherForecastResult(
            success=False,
            error_message=f"Unexpected error: {str(e)}",
        )


def _parse_hourly_forecasts(api_response: dict, live_data: dict | None = None) -> list[HourlyForecast]:
    """Parse hourly forecast data from weerlive.nl API response.

    The weerlive.nl API v2 provides hourly forecast in the 'uur_verw' field at the root level.
    For backwards compatibility, also supports 'uur_verw' inside the live_data object.

    The API provides these fields in the hourly forecast:
    - uur: timestamp in format "DD-MM-YYYY HH:00" (e.g., "02-12-2025 14:00")
    - timestamp: Unix timestamp
    - temp: temperature in °C
    - windkmh: wind speed in km/h
    - windms: wind speed in m/s
    - neersl: precipitation in mm

    Args:
        api_response: The full API response dict (may contain 'uur_verw' at root level).
        live_data: The 'liveweer[0]' data from API response (for backwards compatibility).

    Returns:
        List of HourlyForecast objects.
    """
    forecasts = []

    # Handle different calling patterns:
    # 1. New format: _parse_hourly_forecasts(api_response, live_data) - uur_verw at root level
    # 2. Old format: _parse_hourly_forecasts(live_data) - uur_verw inside live_data (backwards compat)
    if live_data is None and "uur_verw" in api_response:
        # New format with only api_response: uur_verw is at root level
        uur_verw = api_response.get("uur_verw", [])
        # Get humidity/pressure defaults from liveweer[0] if available
        if "liveweer" in api_response and len(api_response["liveweer"]) > 0:
            live_data = api_response["liveweer"][0]
        else:
            live_data = api_response
    elif live_data is None:
        # Old format: first argument is live_data with uur_verw nested inside
        live_data = api_response
        uur_verw = live_data.get("uur_verw", [])
    else:
        # New format with both arguments: check root level first, then fall back
        uur_verw = api_response.get("uur_verw", [])
        if not uur_verw:
            uur_verw = live_data.get("uur_verw", [])

    # Get current values as defaults
    current_humidity = _safe_float(live_data.get("lv", 80))
    current_pressure = _safe_float(live_data.get("luchtd", 1013))

    # Get current datetime for reference
    now = datetime.now()

    for i, hour_data in enumerate(uur_verw):
        if not isinstance(hour_data, dict):
            continue

        # Parse timestamp - the API may provide:
        # 1. "uur" in format "DD-MM-YYYY HH:00" (e.g., "02-12-2025 14:00")
        # 2. "uur" in format "HH:MM" (older format)
        # 3. "timestamp" as Unix timestamp
        uur = hour_data.get("uur", "")
        timestamp_unix = hour_data.get("timestamp")
        
        forecast_dt = None
        
        if timestamp_unix:
            # Use Unix timestamp if available
            try:
                forecast_dt = datetime.fromtimestamp(int(timestamp_unix))
            except (ValueError, OSError, TypeError):
                pass
        
        if forecast_dt is None and uur:
            try:
                # Try DD-MM-YYYY HH:00 format first (e.g., "02-12-2025 14:00")
                if " " in uur and len(uur) > 10:
                    forecast_dt = datetime.strptime(uur, "%d-%m-%Y %H:%M")
                else:
                    # Try HH:MM format (older format)
                    hour, minute = map(int, uur.split(":"))
                    forecast_dt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                    if forecast_dt <= now:
                        forecast_dt += timedelta(days=1)
            except (ValueError, AttributeError):
                pass
        
        if forecast_dt is None:
            # Fall back to incrementing from now
            forecast_dt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=i + 1)

        # Parse forecast values
        temp = _safe_float(hour_data.get("temp", 0))
        
        # Wind speed - try multiple fields:
        # - windkmh: wind speed in km/h (convert to m/s for internal use)
        # - windms: wind speed in m/s
        # - winds: legacy field (assumed m/s)
        wind_speed_ms = 0.0
        if "windkmh" in hour_data:
            # Convert km/h to m/s: divide by 3.6
            wind_speed_ms = _safe_float(hour_data.get("windkmh", 0)) / 3.6
        elif "windms" in hour_data:
            wind_speed_ms = _safe_float(hour_data.get("windms", 0))
        else:
            wind_speed_ms = _safe_float(hour_data.get("winds", 0))

        # Humidity and pressure from current values (API may not provide hourly)
        humidity = _safe_float(hour_data.get("lv", current_humidity))
        pressure = _safe_float(hour_data.get("luchtd", current_pressure))

        # Precipitation
        precip = _safe_float(hour_data.get("neersl", 0))

        # Weather description (may be "image" in new format)
        description = hour_data.get("samenv", hour_data.get("image", ""))

        forecasts.append(HourlyForecast(
            timestamp=forecast_dt,
            temperature=temp,
            wind_speed=wind_speed_ms,
            humidity=humidity,
            pressure=pressure,
            precipitation=precip,
            description=description,
        ))

    return forecasts


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Float value or default.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def convert_forecast_to_scenario_timeslots(
    forecasts: list[HourlyForecast],
    target_temperature: float = 20.0,
) -> list[dict]:
    """Convert weather forecasts to scenario prediction timeslots.

    Args:
        forecasts: List of HourlyForecast objects.
        target_temperature: Target/setpoint temperature to use.

    Returns:
        List of timeslot dictionaries compatible with /api/predictions/scenario.
    """
    timeslots = []

    for forecast in forecasts:
        timeslots.append({
            "timestamp": forecast.timestamp.isoformat(),
            "outdoor_temperature": round(forecast.temperature, 1),
            "wind_speed": round(forecast.wind_speed, 1),
            "humidity": round(forecast.humidity, 1),
            "pressure": round(forecast.pressure, 1),
            "target_temperature": target_temperature,
        })

    return timeslots
