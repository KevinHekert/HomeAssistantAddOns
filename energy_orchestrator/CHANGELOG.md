# Changelog

All notable changes to this add-on will be documented in this file.

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
