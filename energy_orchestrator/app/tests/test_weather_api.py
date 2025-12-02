"""
Tests for weather API module.
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os


@pytest.fixture(scope="module", autouse=True)
def setup_temp_dir():
    """Set up temp directory before module loads."""
    temp_dir = tempfile.mkdtemp()
    os.environ["DATA_DIR"] = temp_dir
    yield temp_dir
    # Cleanup is handled by OS


# Import after setting environment variable
@pytest.fixture
def weather_imports():
    """Import weather module after env is set."""
    # Re-import to use temp directory
    import importlib
    import ha.weather_api as weather_api
    importlib.reload(weather_api)
    return weather_api


class TestWeatherConfig:
    """Test WeatherConfig dataclass."""

    def test_default_values(self, weather_imports):
        """Config has empty defaults."""
        config = weather_imports.WeatherConfig()
        assert config.api_key == ""
        assert config.location == ""

    def test_custom_values(self, weather_imports):
        """Config accepts custom values."""
        config = weather_imports.WeatherConfig(api_key="test_key", location="Amsterdam")
        assert config.api_key == "test_key"
        assert config.location == "Amsterdam"


class TestGetWeatherConfig:
    """Test get_weather_config function."""

    def test_returns_defaults_when_file_missing(self, weather_imports):
        """Returns default config when file doesn't exist."""
        weather_imports._invalidate_cache()
        if weather_imports.WEATHER_CONFIG_FILE_PATH.exists():
            weather_imports.WEATHER_CONFIG_FILE_PATH.unlink()
        
        config = weather_imports.get_weather_config()
        assert config.api_key == ""
        assert config.location == ""

    def test_loads_values_from_file(self, weather_imports):
        """Loads config from file."""
        weather_imports._invalidate_cache()
        weather_imports.WEATHER_CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(weather_imports.WEATHER_CONFIG_FILE_PATH, "w") as f:
            json.dump({"api_key": "my_key", "location": "Rotterdam"}, f)
        
        weather_imports._invalidate_cache()
        config = weather_imports.get_weather_config()
        assert config.api_key == "my_key"
        assert config.location == "Rotterdam"

    def test_ignores_invalid_values(self, weather_imports):
        """Ignores non-string values."""
        weather_imports._invalidate_cache()
        weather_imports.WEATHER_CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(weather_imports.WEATHER_CONFIG_FILE_PATH, "w") as f:
            json.dump({"api_key": 12345, "location": None}, f)
        
        weather_imports._invalidate_cache()
        config = weather_imports.get_weather_config()
        assert config.api_key == ""
        assert config.location == ""

    def test_handles_corrupted_json(self, weather_imports):
        """Returns defaults for corrupted JSON."""
        weather_imports._invalidate_cache()
        weather_imports.WEATHER_CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(weather_imports.WEATHER_CONFIG_FILE_PATH, "w") as f:
            f.write("{invalid json")
        
        weather_imports._invalidate_cache()
        config = weather_imports.get_weather_config()
        assert config.api_key == ""
        assert config.location == ""


class TestSetWeatherConfig:
    """Test set_weather_config function."""

    def test_saves_valid_values(self, weather_imports):
        """Saves valid config values."""
        weather_imports._invalidate_cache()
        success, error = weather_imports.set_weather_config(api_key="test_key", location="Utrecht")
        assert success is True
        assert error is None
        
        config = weather_imports.get_weather_config()
        assert config.api_key == "test_key"
        assert config.location == "Utrecht"

    def test_strips_whitespace(self, weather_imports):
        """Strips whitespace from values."""
        weather_imports._invalidate_cache()
        success, error = weather_imports.set_weather_config(api_key="  key  ", location="  City  ")
        assert success is True
        
        config = weather_imports.get_weather_config()
        assert config.api_key == "key"
        assert config.location == "City"

    def test_partial_update(self, weather_imports):
        """Updates only provided values."""
        weather_imports._invalidate_cache()
        weather_imports.set_weather_config(api_key="key1", location="loc1")
        weather_imports.set_weather_config(location="loc2")
        
        config = weather_imports.get_weather_config()
        assert config.api_key == "key1"
        assert config.location == "loc2"

    def test_rejects_non_string_api_key(self, weather_imports):
        """Rejects non-string API key."""
        success, error = weather_imports.set_weather_config(api_key=12345)
        assert success is False
        assert "string" in error.lower()

    def test_rejects_non_string_location(self, weather_imports):
        """Rejects non-string location."""
        success, error = weather_imports.set_weather_config(location=12345)
        assert success is False
        assert "string" in error.lower()


class TestValidateWeatherApi:
    """Test validate_weather_api function."""

    def test_empty_api_key_fails(self, weather_imports):
        """Empty API key fails validation."""
        success, error, location = weather_imports.validate_weather_api("", "Amsterdam")
        assert success is False
        assert "api key" in error.lower()
        assert location is None

    def test_empty_location_fails(self, weather_imports):
        """Empty location fails validation."""
        success, error, location = weather_imports.validate_weather_api("key", "")
        assert success is False
        assert "location" in error.lower()
        assert location is None

    def test_successful_validation(self, weather_imports):
        """Successful API call returns location name."""
        with patch("ha.weather_api.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "liveweer": [{"plaats": "Amsterdam", "temp": "12.5"}]
            }).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response
            
            success, error, location = weather_imports.validate_weather_api("valid_key", "Amsterdam")
            assert success is True
            assert error is None
            assert location == "Amsterdam"

    def test_invalid_response_fails(self, weather_imports):
        """Invalid API response fails validation."""
        with patch("ha.weather_api.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({"liveweer": []}).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response
            
            success, error, location = weather_imports.validate_weather_api("key", "loc")
            assert success is False
            assert "no weather data" in error.lower()


class TestFetchWeatherForecast:
    """Test fetch_weather_forecast function."""

    def test_no_api_key_returns_error(self, weather_imports):
        """Returns error when API key not configured."""
        weather_imports._invalidate_cache()
        if weather_imports.WEATHER_CONFIG_FILE_PATH.exists():
            weather_imports.WEATHER_CONFIG_FILE_PATH.unlink()
        
        result = weather_imports.fetch_weather_forecast()
        assert result.success is False
        assert "api key" in result.error_message.lower()

    def test_no_location_returns_error(self, weather_imports):
        """Returns error when location not configured."""
        weather_imports._invalidate_cache()
        weather_imports.set_weather_config(api_key="key")
        result = weather_imports.fetch_weather_forecast()
        assert result.success is False
        assert "location" in result.error_message.lower()

    def test_successful_forecast(self, weather_imports):
        """Successful forecast returns hourly data."""
        with patch("ha.weather_api.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "liveweer": [{
                    "plaats": "Amsterdam",
                    "temp": "12.5",
                    "lv": "80",
                    "luchtd": "1013",
                    "uur_verw": [
                        {"uur": "14:00", "temp": "13", "winds": "5", "neersl": "0", "samenv": "Bewolkt"},
                        {"uur": "15:00", "temp": "14", "winds": "6", "neersl": "0.5", "samenv": "Regen"},
                    ]
                }]
            }).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response
            
            weather_imports._invalidate_cache()
            weather_imports.set_weather_config(api_key="key", location="Amsterdam")
            result = weather_imports.fetch_weather_forecast()
            
            assert result.success is True
            assert result.location_name == "Amsterdam"
            assert result.current_temp == 12.5
            assert len(result.hourly_forecasts) == 2

    def test_successful_forecast_weerlive_v2_format(self, weather_imports):
        """Successful forecast with actual Weerlive API v2 format (uur_verw at root level)."""
        with patch("ha.weather_api.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            # Actual API format: uur_verw at root level, windkmh for wind speed
            mock_response.read.return_value = json.dumps({
                "liveweer": [{
                    "plaats": "Amsterdam",
                    "temp": 8.9,
                    "lv": 75,
                    "luchtd": 1007.2,
                }],
                "wk_verw": [
                    {"dag": "02-12-2025", "max_temp": 9, "min_temp": 7}
                ],
                "uur_verw": [
                    {
                        "uur": "02-12-2025 14:00",
                        "timestamp": 1764680400,
                        "temp": 7,
                        "windkmh": 14,
                        "windms": 4,
                        "neersl": 0,
                        "image": "bewolkt"
                    },
                    {
                        "uur": "02-12-2025 15:00",
                        "timestamp": 1764684000,
                        "temp": 8,
                        "windkmh": 18,
                        "windms": 5,
                        "neersl": 0.5,
                        "image": "regen"
                    },
                ]
            }).encode()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response
            
            weather_imports._invalidate_cache()
            weather_imports.set_weather_config(api_key="key", location="Amsterdam")
            result = weather_imports.fetch_weather_forecast()
            
            assert result.success is True
            assert result.location_name == "Amsterdam"
            assert result.current_temp == 8.9
            assert len(result.hourly_forecasts) == 2
            # Verify wind speed conversion: 14 km/h ≈ 3.89 m/s
            assert abs(result.hourly_forecasts[0].wind_speed - (14 / 3.6)) < 0.01
            assert result.hourly_forecasts[0].temperature == 7.0
            assert result.hourly_forecasts[0].humidity == 75.0
            assert result.hourly_forecasts[1].precipitation == 0.5


class TestParseHourlyForecasts:
    """Test _parse_hourly_forecasts function."""

    def test_parses_hourly_data(self, weather_imports):
        """Parses hourly forecast data correctly."""
        live_data = {
            "lv": "75",
            "luchtd": "1015",
            "uur_verw": [
                {"uur": "10:00", "temp": "8", "winds": "3", "neersl": "0"},
                {"uur": "11:00", "temp": "9", "winds": "4", "neersl": "0.2"},
            ]
        }
        
        forecasts = weather_imports._parse_hourly_forecasts(live_data)
        
        assert len(forecasts) == 2
        assert forecasts[0].temperature == 8.0
        assert forecasts[0].wind_speed == 3.0
        assert forecasts[0].humidity == 75.0
        assert forecasts[0].pressure == 1015.0
        assert forecasts[1].precipitation == 0.2

    def test_handles_empty_data(self, weather_imports):
        """Returns empty list for empty data."""
        forecasts = weather_imports._parse_hourly_forecasts({"uur_verw": []})
        assert forecasts == []

    def test_handles_missing_uur_verw(self, weather_imports):
        """Returns empty list when uur_verw is missing."""
        forecasts = weather_imports._parse_hourly_forecasts({})
        assert forecasts == []

    def test_parses_weerlive_v2_format(self, weather_imports):
        """Parses the actual Weerlive API v2 format with uur_verw at root level."""
        # This is the actual format returned by the Weerlive API
        api_response = {
            "liveweer": [{
                "plaats": "Amsterdam",
                "temp": 8.9,
                "lv": 75,
                "luchtd": 1007.2,
            }],
            "uur_verw": [
                {
                    "uur": "02-12-2025 14:00",
                    "timestamp": 1764680400,
                    "temp": 7,
                    "windkmh": 14,
                    "windms": 4,
                    "neersl": 0,
                    "image": "bewolkt"
                },
                {
                    "uur": "02-12-2025 15:00",
                    "timestamp": 1764684000,
                    "temp": 7,
                    "windkmh": 14,
                    "windms": 4,
                    "neersl": 0,
                    "image": "bewolkt"
                },
            ]
        }
        
        live_data = api_response["liveweer"][0]
        forecasts = weather_imports._parse_hourly_forecasts(api_response, live_data)
        
        assert len(forecasts) == 2
        assert forecasts[0].temperature == 7.0
        # windkmh=14 should be converted to m/s: 14 / 3.6 ≈ 3.89
        assert abs(forecasts[0].wind_speed - (14 / 3.6)) < 0.01
        assert forecasts[0].humidity == 75.0
        assert forecasts[0].pressure == 1007.2
        assert forecasts[0].description == "bewolkt"

    def test_parses_windms_field(self, weather_imports):
        """Parses wind speed from windms field (m/s) when windkmh is not present."""
        api_response = {
            "uur_verw": [
                {"uur": "10:00", "temp": "8", "windms": "4", "neersl": "0"},
            ]
        }
        
        forecasts = weather_imports._parse_hourly_forecasts(api_response)
        
        assert len(forecasts) == 1
        assert forecasts[0].wind_speed == 4.0

    def test_parses_datetime_format_with_date(self, weather_imports):
        """Parses datetime in DD-MM-YYYY HH:00 format."""
        api_response = {
            "uur_verw": [
                {"uur": "02-12-2025 14:00", "temp": "8", "windkmh": "10", "neersl": "0"},
            ]
        }
        
        forecasts = weather_imports._parse_hourly_forecasts(api_response)
        
        assert len(forecasts) == 1
        assert forecasts[0].timestamp.year == 2025
        assert forecasts[0].timestamp.month == 12
        assert forecasts[0].timestamp.day == 2
        assert forecasts[0].timestamp.hour == 14


class TestSafeFloat:
    """Test _safe_float function."""

    def test_converts_string(self, weather_imports):
        """Converts string to float."""
        assert weather_imports._safe_float("3.14") == 3.14

    def test_converts_int(self, weather_imports):
        """Converts int to float."""
        assert weather_imports._safe_float(42) == 42.0

    def test_returns_default_for_none(self, weather_imports):
        """Returns default for None."""
        assert weather_imports._safe_float(None) == 0.0
        assert weather_imports._safe_float(None, default=10.0) == 10.0

    def test_returns_default_for_invalid(self, weather_imports):
        """Returns default for invalid values."""
        assert weather_imports._safe_float("invalid") == 0.0
        assert weather_imports._safe_float("", default=5.0) == 5.0


class TestConvertForecastToScenarioTimeslots:
    """Test convert_forecast_to_scenario_timeslots function."""

    def test_converts_forecasts(self, weather_imports):
        """Converts forecasts to scenario timeslots."""
        now = datetime.now()
        forecasts = [
            weather_imports.HourlyForecast(
                timestamp=now,
                temperature=10.0,
                wind_speed=3.0,
                humidity=75.0,
                pressure=1013.0,
            ),
            weather_imports.HourlyForecast(
                timestamp=now + timedelta(hours=1),
                temperature=11.0,
                wind_speed=4.0,
                humidity=70.0,
                pressure=1014.0,
            ),
        ]
        
        timeslots = weather_imports.convert_forecast_to_scenario_timeslots(forecasts)
        
        assert len(timeslots) == 2
        assert timeslots[0]["outdoor_temperature"] == 10.0
        assert timeslots[0]["wind_speed"] == 3.0
        assert timeslots[0]["humidity"] == 75.0
        assert timeslots[0]["pressure"] == 1013.0
        assert timeslots[0]["target_temperature"] == 20.0
        assert timeslots[1]["outdoor_temperature"] == 11.0

    def test_uses_custom_target_temperature(self, weather_imports):
        """Uses custom target temperature."""
        forecasts = [
            weather_imports.HourlyForecast(
                timestamp=datetime.now(),
                temperature=10.0,
                wind_speed=3.0,
                humidity=75.0,
                pressure=1013.0,
            ),
        ]
        
        timeslots = weather_imports.convert_forecast_to_scenario_timeslots(forecasts, target_temperature=21.5)
        
        assert timeslots[0]["target_temperature"] == 21.5

    def test_empty_forecasts(self, weather_imports):
        """Returns empty list for empty forecasts."""
        timeslots = weather_imports.convert_forecast_to_scenario_timeslots([])
        assert timeslots == []


class TestHourlyForecast:
    """Test HourlyForecast dataclass."""

    def test_default_values(self, weather_imports):
        """Has correct default values."""
        forecast = weather_imports.HourlyForecast(
            timestamp=datetime.now(),
            temperature=10.0,
            wind_speed=3.0,
            humidity=75.0,
            pressure=1013.0,
        )
        assert forecast.precipitation == 0.0
        assert forecast.description == ""

    def test_custom_values(self, weather_imports):
        """Accepts custom values."""
        forecast = weather_imports.HourlyForecast(
            timestamp=datetime.now(),
            temperature=10.0,
            wind_speed=3.0,
            humidity=75.0,
            pressure=1013.0,
            precipitation=2.5,
            description="Rainy",
        )
        assert forecast.precipitation == 2.5
        assert forecast.description == "Rainy"


class TestWeatherForecastResult:
    """Test WeatherForecastResult dataclass."""

    def test_success_result(self, weather_imports):
        """Creates successful result."""
        result = weather_imports.WeatherForecastResult(
            success=True,
            hourly_forecasts=[],
            location_name="Amsterdam",
            current_temp=15.0,
        )
        assert result.success is True
        assert result.error_message is None

    def test_error_result(self, weather_imports):
        """Creates error result."""
        result = weather_imports.WeatherForecastResult(
            success=False,
            error_message="Connection failed",
        )
        assert result.success is False
        assert result.error_message == "Connection failed"
