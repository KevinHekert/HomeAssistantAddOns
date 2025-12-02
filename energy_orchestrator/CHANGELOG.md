# Changelog

All notable changes to this add-on will be documented in this file.

## [0.0.0.80] - 2025-12-02

- **Virtual Sensors Implementation**: Implemented Step 4 of sensor configuration feature - Virtual (Derived) Sensors
  - **New Module**: `db/virtual_sensors.py` for virtual sensor management
    - `VirtualSensorDefinition` dataclass for sensor definitions
    - `VirtualSensorOperation` enum: subtract, add, multiply, divide, average
    - `VirtualSensorsConfiguration` class for configuration management
    - `calculate()` method to compute virtual sensor values from two source sensors
    - Persistent JSON storage at `/data/virtual_sensors_config.json`
  - **New API Endpoints**:
    - `GET /api/virtual_sensors/list`: List all virtual sensors with their configurations
    - `POST /api/virtual_sensors/add`: Create a new virtual sensor
    - `DELETE /api/virtual_sensors/<name>`: Delete a virtual sensor
    - `POST /api/virtual_sensors/<name>/toggle`: Enable/disable a virtual sensor
  - **UI Enhancements**:
    - Virtual Sensors section now fully functional
    - Table display showing name, formula, unit, status
    - "Add Virtual Sensor" dialog with prompts for all required fields
    - Enable/Disable buttons for each virtual sensor
    - Delete button with confirmation dialog
    - Reload button to refresh the list
    - Formula display with mathematical symbols (-, +, ×, ÷, avg)
    - Enabled/Disabled status badges
  - **Supported Operations**:
    - `subtract`: sensor1 - sensor2 (e.g., temp_delta = target_temp - indoor_temp)
    - `add`: sensor1 + sensor2 (e.g., total_power = device1 + device2)
    - `multiply`: sensor1 × sensor2 (e.g., energy = power × time)
    - `divide`: sensor1 ÷ sensor2 (with zero-division protection)
    - `average`: (sensor1 + sensor2) / 2
  - **Examples**:
    - Create "temp_delta" = target_temp - indoor_temp (heating demand indicator)
    - Create "outdoor_wind_factor" = wind × outdoor_temp (wind chill indicator)
  - **Note**: Virtual sensor calculations will be integrated into resampling in Step 6

## [0.0.0.79] - 2025-12-02

- **Unit Field Storage and Configuration**: Implemented Step 3 of sensor configuration feature
  - Added `unit` field to `SensorConfig` dataclass
  - Units can now be stored per sensor in the configuration file
  - Unit can override the default unit from `SensorDefinition`
  - **New API Endpoint**: `POST /api/sensors/set_unit`
    - Accepts `category_name` and `unit` parameters
    - Saves unit to sensor configuration
    - Returns updated sensor configuration
  - **Database Model Updates**:
    - `SensorConfig.unit` field added with default empty string
    - `to_dict()` and `from_dict()` methods updated to handle unit
    - `set_unit()` method added to `SensorCategoryConfiguration` class
  - **UI Updates**:
    - Unit input fields are now functional
    - `saveRawSensor()` function now saves both entity ID and unit
    - Displays success message when both are saved successfully
    - Shows warning if entity ID saved but unit failed
  - **Configuration Persistence**:
    - Units are stored in `/data/sensor_category_config.json`
    - Configuration is saved automatically when unit is updated
    - `get_sensors_by_type()` now returns configured unit (or default if not set)
- **Backwards Compatibility**: Existing configurations without unit field will default to empty string

## [0.0.0.78] - 2025-12-02

- **New Sensor Configuration Tab**: Added new "📡 Sensor Configuration" tab in the UI
  - Step 1 of comprehensive sensor configuration and virtual sensor feature
  - Tab positioned between Configuration and Model Training tabs
  - Three main sections created: Raw Sensors, Virtual Sensors, and Feature Stats Configuration
- **Raw Sensors Section**: Interface to view and configure all raw sensors
  - Displays all sensors grouped by type (Usage, Weather, Indoor, Heating)
  - Shows sensor display name, description, entity ID, and unit fields
  - Core sensors displayed with green badge (always enabled)
  - Experimental sensors displayed with orange badge (can be enabled/disabled)
  - Individual Save button for each sensor
  - Uses existing `/api/sensors/category_config` endpoint for loading
  - Uses existing `/api/sensors/set_entity` endpoint for saving
  - `loadRawSensors()` JavaScript function to load and display sensor list
  - `saveRawSensor()` JavaScript function to save individual sensor configuration
- **Virtual Sensors Section** (placeholder for Step 4):
  - Button to add new virtual sensors
  - Reload list button
  - Placeholder message indicating feature will be implemented next
- **Feature Stats Configuration Section** (placeholder for Step 5):
  - Placeholder for time-based statistics configuration
  - Will enable avg_1h, avg_6h, avg_24h, avg_7d generation per sensor
- **UI Improvements**:
  - Added `.action-btn.small` CSS class for inline action buttons
  - Consistent styling with existing UI theme
  - Responsive table layout for sensor configuration
- **Note**: This is the first step of 8-step implementation. Virtual sensors and feature stats functionality will be added in subsequent steps.

## [0.0.0.77] - 2025-12-02

- **Sensor Category Configuration**: Refactored sensor configuration with Core/Experimental categories
  - **Core Sensors** (always enabled, required for prediction):
    - `hp_kwh_total`: Heat Pump kWh Total - Required for training and predictions
    - `outdoor_temp`: Outdoor Temperature - Essential for heating demand prediction
    - `indoor_temp`: Indoor Temperature - Required for setpoint calculations
    - `target_temp`: Target Temperature - Required for heating predictions
    - `wind`: Wind Speed - Affects heat loss and heating demand
    - `humidity`: Humidity - Affects thermal comfort
  - **Experimental Sensors** (optional, can be enabled/disabled via UI):
    - `pressure`: Barometric Pressure - May help predict weather patterns
    - `flow_temp`: Flow Temperature - For advanced heating system monitoring
    - `return_temp`: Return Temperature - For advanced heating system monitoring
    - `dhw_temp`: DHW Temperature - Used to filter DHW cycles
    - `dhw_active`: DHW Active - Helps exclude DHW from training
- **Sensor Configuration UI**: New "Sensor Configuration" card in Configuration tab
  - Sensors grouped by type: Usage, Weather, Indoor, Heating
  - Core sensors shown with green badge (always enabled)
  - Experimental sensors shown with orange badge (toggleable)
  - Entity ID input field for each sensor with Save button
  - Summary showing core/experimental/enabled counts
- **New API Endpoints**:
  - `GET /api/sensors/category_config`: Get sensor configuration with grouping by type
  - `POST /api/sensors/toggle`: Toggle experimental sensors (core sensors cannot be toggled)
  - `POST /api/sensors/set_entity`: Set entity_id for any sensor
  - `GET /api/sensors/definitions`: Get all sensor metadata definitions
- **Configuration Migration**: On first run, sensor settings are automatically migrated from `config.yaml` environment variables to the new JSON-based configuration (`/data/sensor_category_config.json`)
- **Sensor Sync Worker**: Updated to use new sensor category configuration
  - Only enabled sensors are synced
  - Dynamic sensor list based on configuration
- **New Module**: `db/sensor_category_config.py` for sensor category management
  - `SensorDefinition` and `SensorConfig` dataclasses
  - `SensorCategoryConfiguration` class for configuration management
  - Automatic migration from environment variables
  - Persistent JSON storage
- **Tests**: Added 70+ new tests for sensor category functionality
  - 54 tests for sensor category configuration module
  - 16 tests for API endpoints
  - 7 tests for sync_sensor_mappings
- **Note**: Sensor entity IDs in `config.yaml` (e.g., `wind_entity_id`) are now deprecated. They will be used for initial migration only. Configure sensors via the UI instead.

## [0.0.0.76] - 2025-12-02

- **Settings Optimizer**: Added automatic optimization feature to find the best model configuration
  - New "Settings Optimizer" card in the Model Training tab
  - Cycles through different experimental feature combinations
  - Trains both single-step and two-step models for each configuration
  - Finds the configuration with the lowest validation MAPE (%)
  - Current settings are saved before optimization and restored after completion
- **Optimizer UI Features**:
  - "Start Optimization" button to begin the process
  - Live log display showing optimization progress
  - Results table sorted by Val MAPE (best at top)
  - Best result highlighted with star icon (⭐)
  - "Apply Best Settings" button to save the optimal configuration
- **New API Endpoints**:
  - `POST /api/optimizer/run`: Start optimization and get results
  - `GET /api/optimizer/status`: Get current optimizer status
  - `POST /api/optimizer/apply`: Apply the best configuration found
- **New Module**: `ml/optimizer.py` with optimization logic
  - `run_optimization()`: Main optimization function
  - `apply_best_configuration()`: Apply and save optimal settings
  - `OptimizerProgress` and `OptimizationResult` dataclasses
  - Automatic feature combination generation (baseline, individual features, logical groups)
- **Tests**: Added 19 new tests for optimizer functionality
  - Module tests for feature combinations and optimization logic
  - API endpoint tests for run, status, and apply

## [0.0.0.75] - 2025-12-02

- **Fixed Sample Rate Handling in Model Training**: Historical aggregation windows now dynamically adjust based on the configured sample rate
  - Previously, window sizes were hardcoded assuming 5-minute slots (e.g., 12 slots for 1 hour)
  - When using different sample rates (e.g., 15 minutes), the windows were incorrect (12 slots = 3 hours instead of 1 hour)
  - This fix ensures correct historical feature computation for any configured sample rate
- **Dynamic Window Calculations**:
  - Added `_get_slots_per_hour()` helper function to calculate slots based on sample rate
  - `_compute_historical_aggregations()` now accepts `sample_rate_minutes` parameter
  - `_compute_target()` now accepts `sample_rate_minutes` parameter
  - Maximum kWh delta clipping now adjusts based on slot duration (10 kW * hours_per_slot)
  - Heating degree hours calculation uses dynamic hours_per_slot multiplier
- **FeatureDatasetStats Enhancement**: Added `sample_rate_minutes` field to track the sample rate used during training
- **Tests**: Added 2 new tests for verifying correct behavior with different sample rates
  - `test_different_sample_rates` for `_compute_historical_aggregations()`
  - `test_different_sample_rates` for `_compute_target()`
- **Issue Fixed**: Using different resample rates (e.g., 15 minutes) no longer results in weird prediction behavior

## [0.0.0.74] - 2025-12-02

- **Core Feature Expansion**: Moved two features from experimental to core baseline (15 core features total, was 13)
  - `indoor_temp_avg_6h`: 6-hour average indoor temperature - essential for thermal mass tracking
  - `outdoor_temp_avg_1h`: 1-hour outdoor temperature average - essential for near-term weather response
  - These features are now always active and cannot be disabled
- **Improved Training Feedback**: Training output now shows dynamic dataset information
  - Data range: Shows exact start and end timestamps of training data
  - History: Shows total hours of data available (~days)
  - Feature breakdown: Shows count of raw sensor vs calculated features
  - Clearer separation of sections: Dataset, Performance, and Features Used
- **UI Update**: Updated feature count display from 13 to 15 core features
- **Tests**: Added 2 new tests for the new core features (`indoor_temp_avg_6h`, `outdoor_temp_avg_1h`)
- Updated existing tests to reflect new core feature count (13 → 15)

## [0.0.0.73] - 2025-12-02

- **Two-Step Model Feature Display**: Added display of features used by each step in the two-step model
  - Training output now shows features used by Step 1 (Classifier) and Step 2 (Regressor)
  - Each step displays the number of features and the complete feature list
  - Model status check now clarifies that both steps use the same feature set
  - Improved section headers: "Step 1: Classifier" and "Step 2: Regressor" for clarity
  - Fixes issue: "Still can't see what feature-data is used for training models"

## [0.0.0.72] - 2025-12-02

- **Fixed Two-Step Model Training Feedback**: The two-step model training API response now includes top-level `classifier_metrics` and `regressor_metrics` fields for UI compatibility
  - Previously, metrics were only available under `step1_classifier.metrics` and `step2_regressor.metrics`, causing the UI to display "N/A" for all metric values
  - The UI now correctly displays: Accuracy, Precision, Recall, F1 Score for the classifier
  - The UI now correctly displays: Training samples, Validation samples, Train MAE, Val MAE, Val MAPE, Val R² for the regressor
  - The detailed `step1_classifier` and `step2_regressor` objects remain available for advanced API consumers
- **Tests**: Added 2 new tests for the two-step training endpoint API response structure

## [0.0.0.71] - 2025-12-02

- **Feature Verification for Training Models**: Added verification that features displayed are actually used during training
  - Training response now includes `feature_verification` object with:
    - `verified`: Boolean indicating if all features were found in the training dataset
    - `verified_features`: List of features confirmed to be used
    - `missing_in_dataset`: List of features expected but not found in data
  - Training response now includes `feature_categories` showing:
    - `raw_sensor_features`: Features directly from sensors (outdoor_temp, wind, humidity, etc.)
    - `calculated_features`: Derived/aggregated features (outdoor_temp_avg_24h, heating_kwh_last_6h, delta_target_indoor, etc.)
  - Training response now includes `feature_details` with metadata for each feature (category, description, unit, is_calculated)
- **Two-Step Model Step Explanations**: Added clear explanations of what each step does in the two-step model
  - `step1_classifier`: Explains that Step 1 predicts whether heating will be active (on) or inactive (off) for each hour
  - `step2_regressor`: Explains that Step 2 predicts kWh consumption for active hours only (inactive hours = 0 kWh)
  - Each step includes: description, purpose, features_used, feature_count, training_samples, and metrics
- **New Functions in feature_config.py**:
  - `categorize_features()`: Categorizes features as raw sensor vs calculated
  - `get_feature_details()`: Returns detailed metadata for each feature
  - `verify_model_features()`: Verifies model features match dataset features
- **Tests**: Added 7 new tests for feature verification functionality
- **Fixed Weerlive API Parsing**: Fixed parsing of the actual Weerlive API v2 response format
  - The API provides hourly forecast data (`uur_verw`) at the root level of the JSON response, not inside `liveweer[0]`
  - Added support for the actual Weerlive API v2 field names:
    - `windkmh`: Wind speed in km/h (automatically converted to m/s for internal use)
    - `windms`: Wind speed in m/s (alternative field)
    - `timestamp`: Unix timestamp for each hourly forecast
  - Added support for the actual datetime format: "DD-MM-YYYY HH:00" (e.g., "02-12-2025 14:00")
  - Maintains backwards compatibility with older format (uur_verw inside liveweer, winds field, HH:MM format)
- **New Test Cases**: Added 4 new tests for Weerlive API v2 format parsing
  - Test for parsing uur_verw at root level with windkmh field
  - Test for windms field parsing
  - Test for DD-MM-YYYY HH:00 datetime format parsing
  - Integration test for fetch_weather_forecast with full API v2 response
- **Incremental Resampling**: When resampling without flush, the system now starts from the latest resampled slot minus 2x the sample rate, instead of reprocessing all historical data
  - Example: With 5-minute sample rate and latest resampled slot at 12:55, resampling starts from 12:45 (12:55 - 2*5)
  - This significantly improves performance for regular resample operations
  - Existing values that differ will be replaced (idempotent behavior)
  - Use `flush=True` to force full reprocessing from the beginning
- **New Function**: Added `get_latest_resampled_slot_start()` to retrieve the most recent resampled slot timestamp
- **Tests**: Added 9 new tests for incremental resampling behavior
- **Training Data Range Units from Source**: Unit information is now extracted from the actual samples table instead of using hardcoded values
  - `TrainingDataRange` dataclass now includes a `unit` field
  - Units are extracted from the first sample of each category in the resampled data
  - Removed hardcoded `SENSOR_UNITS` dictionary from `app.py`
  - Training data response now displays the actual unit stored with each sensor's data
  - This ensures unit accuracy even for sensors with non-standard units
- Updated `_load_resampled_data()` to include the `unit` column from the database
- Added test for unit extraction from source data
- Updated existing tests to use realistic units in test data

## [0.0.0.70] - 2025-12-02

- **Weather API Integration**: Added integration with weerlive.nl API for weather forecast data
  - New "Weather API Settings" card in Configuration tab for API key and location configuration
  - API credentials are validated before saving to ensure they work
  - Get your free API key at: https://weerlive.nl/delen.php
- **Load Weather Forecast**: Added "🌤️ Load Weather (24h)" button in Scenario-Based Prediction section
  - Fetches upcoming 24-hour weather forecast from weerlive.nl API
  - Automatically populates scenario prediction input with weather data
  - Configurable target temperature for predictions
- **Prediction Storage and Comparison**:
  - Added "💾 Store Prediction" button to save predictions for later comparison
  - New "📊 Stored Predictions" section in Model Training tab
  - Compare stored predictions with actual sensor data when it becomes available
  - View comparison metrics: MAE, MAPE, total predicted vs actual kWh
  - Delete stored predictions when no longer needed
- **New API Endpoints**:
  - `GET /api/weather/config`: Get weather API configuration
  - `POST /api/weather/config`: Save and validate weather API credentials
  - `POST /api/weather/validate`: Validate API credentials without saving
  - `GET /api/weather/forecast`: Fetch weather forecast for next 24 hours
  - `POST /api/predictions/store`: Store a prediction for later comparison
  - `GET /api/predictions/stored`: List all stored predictions
  - `GET /api/predictions/stored/<id>`: Get specific stored prediction
  - `DELETE /api/predictions/stored/<id>`: Delete stored prediction
  - `POST /api/predictions/stored/<id>/compare`: Compare prediction with actual data
- **New Modules**:
  - `ha/weather_api.py`: Weather API integration with weerlive.nl
  - `db/prediction_storage.py`: Prediction storage and comparison functionality
- **Tests**: Added 50 new tests for weather API and prediction storage functionality

## [0.0.0.69] - 2025-12-02

- **Scenario Prediction Uses Two-Step When Enabled**: The simplified scenario prediction endpoint now automatically uses the two-step model when enabled
  - When two-step prediction is enabled in Feature Configuration and the two-step model is trained, `/api/predictions/scenario` will use the two-step approach
  - Response includes `is_active` and `activity_probability` for each timeslot when using two-step
  - Response includes summary with active/inactive hour counts
  - UI displays active/inactive status with 🔥/❄️ icons for each hour
  - Falls back to single-step model if two-step is not available
- **UI Improvements for Two-Step Scenario Predictions**:
  - Status message indicates "(using two-step prediction)" when enabled
  - Results table shows "Active" column with status icons (🔥 Active / ❄️ Inactive)
  - Results table shows "Activity Prob." column with classifier probability
  - Summary box shows count of active vs inactive hours
- Added 4 new tests for two-step scenario prediction functionality

## [0.0.0.68] - 2025-12-02

- **Two-Step Prediction UI Improvements**: Redesigned the two-step prediction feature for better visibility and clarity
  - Moved two-step prediction toggle to its own dedicated card in the Configuration tab
  - Card styled as an experimental feature similar to other experimental options
  - Added detailed description explaining the two-step approach (classifier + regressor)
  - Shows current model status (available/not trained) and activity threshold
- **Two-Step Model Training Section**: Added dedicated training section in Model Training tab
  - New "Two-Step Model Training" card with experimental badge
  - Displays training statistics: activity threshold, active samples, inactive samples, classifier accuracy
  - Shows detailed classifier metrics (accuracy, precision, recall, F1 score)
  - Shows detailed regressor metrics (MAE, MAPE, R²) for active hours only
  - Clear feedback showing how many hours were classified as active vs inactive
  - Training data range table showing sensor values used
- **Improved Training Feedback**: When training the two-step model, users now clearly see:
  - The computed activity threshold (minimum kWh to consider heating "active")
  - Number of active vs inactive samples in the training data
  - Classifier performance metrics
  - Regressor performance on active hours only

## [0.0.0.67] - 2025-12-02

- **Historical Day Prediction Timestamps**: When loading historical data for scenario predictions, timestamps are now adjusted to be 2 days after today (day after tomorrow)
  - The `scenario_format` in `/api/examples/historical_day/<date>` now contains future timestamps for prediction
  - Original historical timestamps are preserved in `hourly_data` for comparison purposes
  - This allows comparing predictions with actual historical data while using valid future timestamps for the prediction API
- Added test for timestamp adjustment in historical day scenario format

## [0.0.0.66] - 2025-12-02

- **Sync Configuration UI**: Added UI controls to configure sensor sync settings
  - New "Sync Configuration" card in the Configuration tab
  - Backfill Days: How many days to look back when no samples exist (1-365, default: 14)
  - Sync Window Size: Size of each sync window in days (1-30, default: 1)
  - Sensor Sync Interval: Wait time in seconds between syncing sensors (1-3600, default: 1)
  - Loop Interval: Wait time in seconds between sync loop iterations (1-3600, default: 1)
  - Configuration is persisted to `/data/sync_config.json`
- **Two-Step Prediction Toggle in UI**: Added checkbox in Feature Configuration to enable/disable two-step prediction
  - Visual toggle with description of the two-step approach
  - Status is loaded on page load
- **New API Endpoints**:
  - `GET /api/sync_config`: Get current sync configuration with limits
  - `POST /api/sync_config`: Update sync configuration (partial updates supported)
- **New Module**: `db/sync_config.py` for persistent sync configuration storage
  - `SyncConfig` dataclass with all sync settings
  - `get_sync_config()`, `set_sync_config()` for full configuration access
  - Individual getters: `get_backfill_days()`, `get_sync_window_days()`, `get_sensor_sync_interval()`, `get_sensor_loop_interval()`
- **Code Changes**:
  - `ha_api.py`: Now uses `get_backfill_days()` and `get_sync_window_days()` from sync_config module
  - `sensors.py`: Now uses `get_sensor_sync_interval()` and `get_sensor_loop_interval()` from sync_config module
- **Tests**: Added 23 new tests for sync configuration module

## [0.0.0.65] - 2025-12-02

- **Removed UI Sections**: Removed the following UI sections from the Model Training tab:
  - "🔮 Single Slot Prediction" - removed card and functionality
  - "📅 Full Day Prediction (24h)" - removed card and functionality
- **Removed API Endpoints**:
  - `GET /api/examples/single_slot` - no longer available
  - `GET /api/examples/full_day` - no longer available
- **Note**: The `/api/predictions/heating_demand_profile` endpoint is still available for external use
- Removed related tests (6 tests removed)

## [0.0.0.64] - 2025-12-02

- **Preserve Actual Sensor Timestamps**: Sample timestamps now reflect the actual timestamps from Home Assistant
  - Removed 5-second alignment that rounded all timestamps to nearest 5-second boundary
  - Timestamps like `2025-11-21 06:52:27` are now stored as-is instead of being rounded to `06:52:25`
  - Only microseconds are stripped (for storage efficiency), seconds are preserved
  - This provides accurate time information for debugging and data analysis
- **Breaking Change**: New samples will have actual timestamps, but existing samples remain with rounded timestamps
  - To get consistent timestamps, users may want to re-sync historical data after updating
- Updated tests to reflect new timestamp preservation behavior

## [0.0.0.63] - 2025-12-02

- **Two-Step Heat Pump Prediction (Experimental)**: Added new two-step prediction approach for better accuracy
  - Step 1: Classifier predicts whether heating will be active or inactive in a given hour
  - Step 2: Regressor predicts kWh consumption only for active hours (inactive = 0 kWh)
  - Solves overestimation when pump is off and underestimation during heavy heating
- **Automatic Threshold Detection**: Activity threshold is automatically computed from training data
  - Uses 5th percentile of positive kWh values as threshold
  - Minimum threshold of 0.01 kWh to filter noise
  - Threshold is stored with model and reused for predictions
  - No manual configuration required by end users
- **New API Endpoints**:
  - `GET /api/features/two_step_prediction`: Get two-step prediction configuration status
  - `POST /api/features/two_step_prediction`: Enable/disable two-step prediction mode
  - `POST /api/train/two_step_heating_demand`: Train the two-step model (classifier + regressor)
  - `GET /api/model/two_step_status`: Get two-step model status and threshold info
  - `POST /api/predictions/two_step_scenario`: Make predictions using two-step approach
- **New Module**: `ml/two_step_model.py` with complete implementation
  - `TwoStepHeatingDemandModel` class with classifier and regressor
  - `TwoStepPrediction` dataclass with is_active, predicted_kwh, activity_probability
  - `train_two_step_heating_demand_model()` for training both models
  - `predict_two_step_scenario()` for scenario-based predictions
- **Feature Configuration Updates**:
  - Added `two_step_prediction_enabled` flag to FeatureConfiguration
  - New methods: `enable_two_step_prediction()`, `disable_two_step_prediction()`, `is_two_step_prediction_enabled()`
  - Configuration persists across restarts
- **Tests**: Added 27 new tests for two-step prediction functionality
  - Threshold computation tests
  - Classifier and regressor training tests
  - Prediction tests for active/inactive hours
  - Model persistence tests
  - Feature configuration tests

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
