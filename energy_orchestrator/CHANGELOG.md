# Changelog

All notable changes to this add-on will be documented in this file.

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
