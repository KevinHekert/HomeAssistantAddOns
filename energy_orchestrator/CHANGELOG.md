# Changelog

All notable changes to this add-on will be documented in this file.

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
