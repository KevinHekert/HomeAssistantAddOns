# Changelog

All notable changes to this add-on will be documented in this file.

## [0.0.0.62] - 2025-12-02

- **Resampling Interval UI Configuration**: Moved sample rate configuration from config.yaml to UI
  - Sample rate can now be changed directly from the Configuration tab in the UI
  - New dropdown selector for sample rate (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60 minutes)
  - Sample rate is persisted to `/data/resample_config.json` for persistence across restarts
  - When sample rate is changed, the "Flush existing data" checkbox is automatically checked
- **API Changes**:
  - Added `POST /api/sample_rate` endpoint to update sample rate from UI
  - Added `set_sample_rate_minutes()` function to persist sample rate configuration
  - Sample rate is now read from persistent JSON config instead of environment variable
- **Config Changes**:
  - Removed `sample_rate_minutes` from add-on configuration options
  - Removed `SAMPLE_RATE_MINUTES` environment variable from run.sh
- Added tests for new sample rate persistence functionality

## [0.0.0.61] - 2025-12-02

- **Improved Training Data Table**: Enhanced training data display with all sensor categories
  - Now shows all sensor categories (outdoor_temp, wind, humidity, pressure, indoor_temp, target_temp, dhw_temp, hp_kwh_total, dhw_active) instead of just dhw_temp and hp_kwh_total
  - For hp_kwh_total, displays the delta (energy consumed during training) instead of raw cumulative values
  - Added Delta column showing the difference between first and last values for all sensors
  - hp_kwh_delta is highlighted in green to emphasize the actual energy consumption
  - Units are now displayed for each sensor value
- **API Changes**:
  - `/api/train/heating_demand` now returns `training_data` with all sensor categories
  - Each sensor category includes `first`, `last`, and `unit` fields
  - hp_kwh_total additionally includes `delta` field showing energy consumed during training
  - Added `sensor_ranges` dict to `FeatureDatasetStats` for tracking all sensor categories
  - Added `hp_kwh_delta` field to `FeatureDatasetStats` for energy consumption delta
  - Legacy fields `dhw_temp_range` and `hp_kwh_total_range` maintained for backward compatibility
- Added 4 new tests for sensor ranges and hp_kwh_delta functionality

## [0.0.0.60] - 2025-12-02

- **Dark/Light Mode Toggle**: Added theme toggle button in header
  - Uses CSS variables for consistent theming across light and dark modes
  - Theme preference is saved to localStorage
  - Defaults to dark mode
- **Tabbed Interface**: Reorganized UI into three tabs
  - Configuration tab: Data resampling and feature configuration
  - Model Training tab: Model training, predictions, and scenario testing
  - Sensor Information tab: Sensor data overview
- **Training Data Table**: Added table showing first/last values when training model
  - Displays dhw_temp and hp_kwh_total ranges
  - API endpoint updated to return `training_data` with first/last values
  - New `TrainingDataRange` dataclass for capturing sensor ranges
- **Removed Wind Card**: Removed test wind speed card from UI

## [0.0.0.59] - 2025-12-02

- **Flush Resample Table on Sample Rate Change**: Added ability to flush existing resampled data before resampling
  - New `flush_resampled_samples()` function to clear the resampled_samples table
  - `/resample` endpoint now accepts optional `flush` parameter in request body
  - When sample rate changes, existing data (computed with different interval) should be flushed
  - `ResampleStats` now includes `table_flushed` field to indicate if flush was performed
  - UI now shows a "Flush existing data" checkbox in the Data Resampling section
- **Dynamic Sample Rate Display in UI**: UI now shows configured sample rate instead of hardcoded "5-minute"
  - Sample rate is loaded from `/api/sample_rate` endpoint on page load
  - Resampling status message includes the sample rate used
- Added tests for flush functionality

## [0.0.0.58] - 2025-12-02

- **Configurable Sample Rate**: Added ability to configure the sample rate for data resampling
  - New `sample_rate_minutes` configuration option (1-60 minutes, default: 5)
  - Allows training models with different time granularity for different use cases
  - kWh usage is correctly calculated for the configured timeframe
  - `/resample` endpoint now accepts optional `sample_rate_minutes` in request body
  - New `/api/sample_rate` endpoint to get current sample rate configuration
  - `ResampleStats` now includes `sample_rate_minutes` field
  - Added 17 new tests for configurable sample rate functionality

## [0.0.0.57] - 2025-12-02

- **Feature Set Configuration**: Complete overhaul of the heat pump consumption model feature set
  - Defined **13 core baseline features** that are always active and cannot be disabled
  - Moved optional features to **experimental status** (disabled by default, toggleable via UI)
  - All features now have complete metadata (name, category, description, unit, time_window, is_core)
  
- **New Core Baseline Features**:
  - `heating_kwh_last_1h`: 1-hour heating energy consumption (was missing)
  - `delta_target_indoor`: Derived feature showing difference between target and indoor temperature
  - `wind` and `humidity` now explicitly in baseline (always required)
  
- **Timezone Configuration**:
  - `hour_of_day` feature now uses configurable IANA timezone (default: Europe/Amsterdam)
  - All timestamps stored in UTC and converted to local time for hour_of_day
  - UI allows timezone selection from common timezones
  
- **Feature Configuration UI**:
  - New "Feature Configuration" section showing all features grouped by category
  - Core features (green badges) are always active with disabled checkboxes
  - Experimental features (orange badges) can be toggled on/off
  - Timezone selector for hour_of_day feature
  - Feature stats showing core count and active feature count
  
- **New API Endpoints**:
  - `GET /api/features/config`: Get current feature configuration
  - `POST /api/features/toggle`: Enable/disable experimental features
  - `POST /api/features/timezone`: Set timezone for time features
  - `GET /api/features/metadata`: Get feature metadata for documentation
  
- **Tests**: Added 38 new tests for feature configuration module
- **Documentation**: Updated README with new feature engineering documentation

![Feature Configuration UI](https://github.com/user-attachments/assets/8bf955e4-cee7-4b84-9a7d-47fdf464c354)

## [0.0.0.56] - 2025-12-01

- **Version Bump**: Preparing for feature set refactoring of heat pump consumption model

## [0.0.0.55] - 2025-12-01

- **Load Historical Day Examples**: Added ability to load historical days from 5-minute resampled data as scenario examples
  - New `/api/examples/available_days` endpoint returns list of available days (excluding first and last day)
  - New `/api/examples/historical_day/<date>` endpoint returns hourly averaged data for a specific day
  - UI dropdown to select historical day in the Scenario-Based Prediction section
- **Compare Predictions with Actual Data**: When a historical day is selected, the prediction table shows comparison
  - Added second column with actual kWh values from historical data
  - Added delta (difference) and percentage columns
  - Bar chart shows predicted vs actual values side by side
  - Summary shows Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE)
- Added 19 new tests for historical day functionality
- Internal: Added `get_available_historical_days()` function to query available days
- Internal: Added `get_historical_day_hourly_data()` function to compute hourly averages

## [0.0.0.54] - 2025-12-01

- **Simplified Scenario API**: Added new `/api/predictions/scenario` endpoint that accepts human-readable inputs
  - Users can now send simple weather forecast data (outdoor_temperature, wind_speed, humidity, pressure) and setpoint schedule (target_temperature)
  - All low-level model features (time features, historical aggregations) are computed internally
  - Timestamps must be in the future; past timestamps are rejected with clear validation errors
- Added `/api/examples/scenario` endpoint to get pre-filled 24-hour scenario example
- Added new UI section "Scenario-Based Prediction (Simplified)" with bar chart and table visualization
- Internal: Added validation functions for simplified scenario input
- Internal: Added conversion function to translate simplified inputs to model features
- Added 29 new tests for new simplified scenario functionality
- Updated documentation with new simplified API endpoints

## [0.0.0.53] - 2025-12-01

- **Enhanced model training**: Model now uses ALL available historical data for training (no artificial limit)
- Added function to compute historical aggregations from user-provided scenario features
- Added API endpoint `/api/predictions/enrich_scenario` to help users prepare prediction requests
- Added API endpoint `/api/predictions/compare_actual` to compare predictions with actual historical data
- Added API endpoint `/api/predictions/validate_start_time` to validate prediction start times
- Added time range information to dataset statistics (data_start_time, data_end_time, available_history_hours)
- Predictions must start at next/coming hour for accurate historical feature computation
- Test data can be compared using 5-minute average records to see delta between model and actual values
- Added 25 new tests for new functionality
- Updated documentation with new API endpoints and enhanced feature descriptions

## [0.0.0.52] - 2025-12-01

- Added pre-filled example fields in UI for single slot and full day heat pump usage calculation
- Added sensor information section in UI showing first and last datetime per sensor
- Added API endpoints: /api/examples/single_slot, /api/examples/full_day, /api/sensors/info
- Added results display with table and bar chart for predictions
- Added 12 new tests for new API endpoints and get_sensor_info function
- Improved UI with editable JSON input fields for predictions
- Added comprehensive technical and functional documentation (README.md)
- Documented all calculations: time-weighted averaging, historical aggregations, feature engineering
- Documented model storage location (`/data/heating_demand_model.joblib`)
- Added complete API reference with request/response examples
- Added usage examples for Python and Home Assistant automations
- Documented database schema and architecture

## [0.0.0.51] - 2025-12-01

- Fixed resampling to properly return statistics (ResampleStats) with slots_processed, slots_saved, slots_skipped
- Added UI buttons for model training and status checking
- Improved resample endpoint to show detailed statistics after completion
- Updated UI to display resampling statistics (time range, categories, slot counts)
- Added 4 new tests for ResampleStats return values

## [0.0.0.50] - 2025-12-01

- Fixed sensor import halting when data gaps exceed 24 hours
- Sync now fast-forwards through gaps in historical data by checking subsequent 24-hour windows
- Uses max(latest_sample_timestamp, sync_status.last_attempt) to ensure progress through data gaps
- Added tests for DWH sensor scenarios with large gaps in history data

## [0.0.0.49] - 2025-12-01

- Added comprehensive unit tests for sync_state module
- Added unit tests for Flask API endpoints (training, prediction, model status)
- Created GitHub Actions workflow to run tests on push and pull requests
- Improved test coverage from 102 to 120 tests

## [0.0.0.48] - 2025-12-01

- Added support for binary sensor states: map on/off and true/false to 1.0/0.0 for database storage

## [0.0.0.47] - 2025-12-01

- Reduced logging verbosity by changing sample save/update messages from INFO to DEBUG level
