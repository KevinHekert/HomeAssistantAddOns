# Energy Orchestrator

A Home Assistant add-on for energy optimization using machine learning. This add-on collects sensor data from your heating system, trains a predictive model, and provides heating demand forecasts.

## Overview

The Energy Orchestrator is designed to help optimize energy usage in your home by:

1. **Collecting sensor data** from Home Assistant (temperature, weather, heat pump metrics)
2. **Resampling raw data** into uniform 5-minute time slots for analysis
3. **Training a machine learning model** to predict heating energy demand
4. **Providing predictions** via API endpoints for integration with automation

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Data Collection & Processing](#data-collection--processing)
- [Machine Learning Model](#machine-learning-model)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
  - [Model Storage](#model-storage)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Technical Details](#technical-details)

---

## Installation

1. Add this repository to Home Assistant's Add-on Store
2. Install the Energy Orchestrator add-on
3. Configure the required sensors (see [Configuration](#configuration))
4. Start the add-on

**Requirements:**
- MariaDB add-on (or external MySQL database)
- Home Assistant Supervisor API access

---

## Configuration

Configure the add-on with your sensor entity IDs:

| Option | Description | Required |
|--------|-------------|----------|
| `db_host` | Database hostname | Yes |
| `db_user` | Database username | Yes |
| `db_password` | Database password | Yes |
| `db_name` | Database name | Yes |
| `wind_entity_id` | Wind speed sensor | No |
| `outdoor_temp_entity_id` | Outdoor temperature sensor | No |
| `flow_temp_entity_id` | Flow temperature sensor | No |
| `return_temp_entity_id` | Return temperature sensor | No |
| `humidity_entity_id` | Humidity sensor | No |
| `pressure_entity_id` | Barometric pressure sensor | No |
| `hp_kwh_total_entity_id` | Heat pump total kWh sensor | No |
| `dhw_temp_entity_id` | Domestic hot water temperature | No |
| `indoor_temp_entity_id` | Indoor temperature sensor | No |
| `target_temp_entity_id` | Target/setpoint temperature | No |
| `dhw_active_entity_id` | DHW active binary sensor | No |

**Example configuration:**

```yaml
db_host: core-mariadb
db_user: energy_orchestrator
db_password: your_password
db_name: energy_orchestrator
wind_entity_id: sensor.knmi_windsnelheid
outdoor_temp_entity_id: sensor.smile_outdoor_temperature
indoor_temp_entity_id: sensor.anna_temperature
target_temp_entity_id: sensor.anna_setpoint
hp_kwh_total_entity_id: sensor.extra_total
```

---

## Data Collection & Processing

### Sensor Data Synchronization

The add-on continuously syncs sensor history from Home Assistant:

1. **Historical backfill**: On first run, syncs up to 200 days of historical data
2. **Continuous updates**: Periodically fetches new data from Home Assistant's history API
3. **Gap handling**: Automatically fast-forwards through periods with missing data

Data is stored in the `samples` table with:
- `entity_id`: The Home Assistant entity
- `timestamp`: Sample timestamp (aligned to 5-second boundaries for consistent storage)
- `value`: Numeric sensor value
- `unit`: Unit of measurement

### Resampling to 5-Minute Slots

Raw samples (stored at 5-second granularity) are aggregated into uniform 5-minute time slots using **time-weighted averaging**:

1. **Category mapping**: Each sensor is mapped to a logical category (e.g., `outdoor_temp`, `wind`)
2. **Time-weighted calculation**: Uses last-known-value (zero-order hold) interpolation
3. **Complete slots only**: A slot is only saved if ALL configured categories have valid data

**Time-Weighted Average Formula:**

For a 5-minute window `[t₀, t₁)`:

```
avg = Σ(value_i × duration_i) / total_duration
```

Where:
- `value_i` is the value during interval `i`
- `duration_i` is the duration that value was held
- `total_duration` is `t₁ - t₀` (300 seconds)

The resampled data is stored in `resampled_samples`:
- `slot_start`: Start of the 5-minute slot
- `category`: Logical category name
- `value`: Time-weighted average value
- `unit`: Unit of measurement

---

## Machine Learning Model

### Feature Engineering

The model uses only **exogenous variables** (external factors) as input features. Heat pump outputs like flow temperature and power are explicitly excluded to prevent data leakage.

All features are derived from **5-minute averaged samples**:
- 1 hour  = last **12** samples
- 6 hours = last **72** samples
- 24 hours = last **288** samples

#### Core Baseline Features (13 features)

The model uses **exactly 13 core baseline features** by default. These features are always active and cannot be disabled. They represent the key physical drivers of heating energy consumption.

| # | Feature | Category | Description | Unit | Window |
|---|---------|----------|-------------|------|--------|
| 1 | `outdoor_temp` | Weather | Latest 5-minute outdoor temperature | °C | - |
| 2 | `outdoor_temp_avg_24h` | Weather | 24-hour outdoor temp average (last 288 samples) | °C | 24h |
| 3 | `wind` | Weather | Latest 5-minute wind speed or intensity | m/s | - |
| 4 | `humidity` | Weather | Latest 5-minute outdoor relative humidity | % | - |
| 5 | `indoor_temp` | Indoor | Latest 5-minute indoor temperature | °C | - |
| 6 | `indoor_temp_avg_24h` | Indoor | 24-hour average indoor temp (building mass/thermal history) | °C | 24h |
| 7 | `target_temp` | Control | Latest 5-minute heating target setpoint | °C | - |
| 8 | `target_temp_avg_6h` | Control | 6-hour average heating target setpoint | °C | 6h |
| 9 | `heating_kwh_last_1h` | Usage | Heating energy in last 1 hour | kWh | 1h |
| 10 | `heating_kwh_last_6h` | Usage | Heating energy in last 6 hours | kWh | 6h |
| 11 | `heating_kwh_last_24h` | Usage | Heating energy in last 24 hours | kWh | 24h |
| 12 | `hour_of_day` | Time | Local hour (0-23), using configured timezone | hour | - |
| 13 | `delta_target_indoor` | Control | Difference: target_temp - indoor_temp | °C | - |

> **Note:** The baseline feature set is designed to be small, strictly defined, and physically meaningful. It focuses on the key drivers of energy usage, reduces overfitting on small datasets, and remains easy to extend with optional features.

#### Experimental Features (Optional)

The following features are **experimental** and disabled by default. They can be enabled/disabled via the UI without breaking the training or prediction pipelines.

| Feature | Category | Description | Unit | Window |
|---------|----------|-------------|------|--------|
| `pressure` | Weather | Latest 5-minute barometric pressure | hPa | - |
| `outdoor_temp_avg_1h` | Weather | 1-hour outdoor temp average | °C | 1h |
| `outdoor_temp_avg_6h` | Weather | 6-hour outdoor temp average | °C | 6h |
| `outdoor_temp_avg_7d` | Weather | 7-day outdoor temp average* | °C | 7d |
| `indoor_temp_avg_6h` | Indoor | 6-hour average indoor temp | °C | 6h |
| `target_temp_avg_24h` | Control | 24-hour average heating setpoint | °C | 24h |
| `heating_kwh_last_7d` | Usage | Heating energy in last 7 days* | kWh | 7d |
| `heating_degree_hours_24h` | Usage | Heating degree hours over 24h | °C·h | 24h |
| `heating_degree_hours_7d` | Usage | Heating degree hours over 7 days* | °C·h | 7d |
| `day_of_week` | Time | Day of week (0=Monday, 6=Sunday) | day | - |
| `is_weekend` | Time | 1 if Saturday/Sunday, else 0 | boolean | - |
| `is_night` | Time | 1 if hour is 23:00-06:59, else 0 | boolean | - |

\* 7-day features require at least 7 days (168 hours / 2016 five-minute slots) of history.

#### Time Zone Configuration

- All raw timestamps are stored in **UTC**.
- For `hour_of_day`, timestamps are converted to a **configurable IANA timezone**.
- Default timezone: `Europe/Amsterdam` (with correct daylight saving handling).
- Configure the timezone via the UI or API.

#### Feature Metadata

All features (baseline + experimental) have associated metadata:
- `name`: Feature name
- `category`: weather | indoor | control | usage | time
- `description`: Human-readable description
- `unit`: Unit of measurement (°C, kWh, %, m/s, etc.)
- `time_window`: none | 1h | 6h | 24h | 7d
- `is_core`: True for baseline features

Use the `/api/features/metadata` endpoint to retrieve this information.

### Making Predictions

Predictions are always for future time periods. The request should contain:

- **User-provided inputs (predicted/forecasted values):**
  - Predicted outdoor temperature (hourly)
  - Predicted wind speed (hourly)
  - Predicted humidity (hourly)
  - Predicted pressure (hourly)
  - Hourly set temperature (target/setpoint)

- **System-derived inputs (optional, or use `/api/predictions/enrich_scenario`):**
  - Historical temperature averages (can be derived from forecast)
  - Heating degree hours
  - Time features (hour_of_day, day_of_week, etc.)

**Important:** Predictions should start at the next or coming hour. This ensures the system has the latest historical data available for accurate predictions.

#### Target Variable

The target is the **heating energy demand** over the next hour:

```
target_heating_kwh_1h = hp_kwh_total[t+12] - hp_kwh_total[t]
```

Where `t+12` means 12 slots (1 hour) into the future.

**Filtering rules:**
- DHW (domestic hot water) slots are excluded from training
- Implausible values (negative or >20 kWh/hour) are filtered out

### Model Training

The model uses **Gradient Boosting Regression** from scikit-learn:

```python
GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
)
```

**Training process:**

1. Load resampled samples from database
2. Compute historical aggregation features
3. Compute target (forward-looking kWh delta)
4. Add time features
5. Filter rows with missing values
6. Time-based train/validation split (80% train, 20% validation)
7. Fit model and compute metrics
8. Save model to disk

**Evaluation metrics:**
- **MAE (Mean Absolute Error)**: Average absolute prediction error in kWh
- **MAPE (Mean Absolute Percentage Error)**: Percentage error
- **R² (Coefficient of Determination)**: Explained variance ratio

### Model Storage

The trained model is persisted to disk for reuse across restarts.

**Storage location:**

```
/data/heating_demand_model.joblib
```

The `/data` directory is the persistent storage volume for Home Assistant add-ons.

**Stored data includes:**
- Trained scikit-learn model object
- Feature names (list of features in expected order)
- Training timestamp
- Training metrics (MAE, MAPE, R²)

**Environment variable:**
You can override the model directory by setting `MODEL_DIR`:

```
MODEL_DIR=/custom/path
```

The model file name is always `heating_demand_model.joblib`.

### Two-Step Prediction (Experimental)

The two-step prediction approach addresses a common issue where the single-regressor model:
- **Overestimates** when the heat pump is off (predicting kWh when there should be 0)
- **Underestimates** during heavy heating (averaging with inactive hours)

#### How It Works

1. **Step 1 - Classifier**: Predicts whether the hour will be "active" (heating on) or "inactive" (no heating)
2. **Step 2 - Regressor**: For active hours only, predicts how many kWh will be used. Inactive hours automatically return 0 kWh.

#### Automatic Threshold Detection

The system automatically determines what counts as "active" vs "inactive":

- Analyzes the distribution of target kWh values during training
- Uses the **5th percentile** of positive kWh values as the threshold
- Minimum threshold of **0.01 kWh** (10 Wh) to filter noise/standby consumption
- Threshold is stored with the model and reused for predictions

**No manual configuration required** - the threshold is computed automatically from your data.

#### Enabling Two-Step Prediction

```http
POST /api/features/two_step_prediction
Content-Type: application/json

{"enabled": true}
```

#### Training the Two-Step Model

```http
POST /api/train/two_step_heating_demand
```

This trains both the classifier and regressor. Response includes:
- `threshold`: Computed activity threshold and sample counts
- `classifier_metrics`: Accuracy, precision, recall, F1 score
- `regressor_metrics`: MAE, MAPE, R² (on active samples only)

#### Making Two-Step Predictions

```http
POST /api/predictions/two_step_scenario
Content-Type: application/json

{
  "timeslots": [
    {
      "timestamp": "2024-01-15T14:00:00",
      "outdoor_temperature": 5.0,
      "wind_speed": 3.0,
      "humidity": 75.0,
      "pressure": 1013.0,
      "target_temperature": 20.0
    }
  ]
}
```

Response includes for each hour:
- `is_active`: Whether heating is predicted to be active
- `predicted_kwh`: Predicted consumption (0 for inactive hours)
- `activity_probability`: Classifier confidence (0-1)

#### Model Storage

The two-step model is stored separately:

```
/data/heating_demand_two_step_model.joblib
```

Contains:
- Trained classifier (GradientBoostingClassifier)
- Trained regressor (GradientBoostingRegressor)
- Activity threshold (kWh)
- Feature names and training timestamp

---

## API Reference

### Trigger Data Resampling

Resample raw sensor samples into 5-minute slots.

```http
POST /resample
```

**Response:**

```json
{
  "status": "success",
  "message": "Resampling completed successfully",
  "stats": {
    "slots_processed": 1440,
    "slots_saved": 1420,
    "slots_skipped": 20,
    "categories": ["outdoor_temp", "indoor_temp", "wind"],
    "start_time": "2024-01-01T00:00:00",
    "end_time": "2024-01-06T00:00:00"
  }
}
```

---

### Train Heating Demand Model

Train a new heating demand prediction model.

```http
POST /api/train/heating_demand
```

**Response (success):**

```json
{
  "status": "success",
  "message": "Model trained successfully",
  "metrics": {
    "train_samples": 8000,
    "val_samples": 2000,
    "train_mae_kwh": 0.1234,
    "val_mae_kwh": 0.1567,
    "val_mape_pct": 12.34,
    "val_r2": 0.8901,
    "features": ["outdoor_temp", "wind", "hour_of_day", ...]
  },
  "dataset_stats": {
    "total_slots": 12000,
    "valid_slots": 10000,
    "features_used": ["outdoor_temp", "wind", ...],
    "has_7d_features": true,
    "data_start_time": "2024-01-01T00:00:00",
    "data_end_time": "2024-03-15T00:00:00",
    "available_history_hours": 1776.0
  }
}
```

> **Note:** The model uses ALL available historical data for training. There is no artificial limit (e.g., 7 days). If your model is trained for over a year, a wide variation of days/weather conditions will be available for better predictions.

**Response (insufficient data):**

```json
{
  "status": "error",
  "message": "Insufficient data for training",
  "stats": {
    "total_slots": 50,
    "valid_slots": 30,
    "dropped_missing_features": 10,
    "dropped_missing_target": 10
  }
}
```

---

### Get Model Status

Check if a trained model is available.

```http
GET /api/model/status
```

**Response (model available):**

```json
{
  "status": "available",
  "features": ["outdoor_temp", "wind", "humidity", "hour_of_day", ...],
  "training_timestamp": "2024-01-15T10:30:00"
}
```

**Response (no model):**

```json
{
  "status": "not_available",
  "message": "No trained model available"
}
```

---

### Predict Heating Demand Profile

Predict heating demand for multiple future time slots.

```http
POST /api/predictions/heating_demand_profile
```

**Request body:**

```json
{
  "timeslots": ["2024-01-15T12:00:00", "2024-01-15T13:00:00"],
  "scenario_features": [
    {
      "outdoor_temp": 5.0,
      "wind": 3.0,
      "humidity": 80.0,
      "pressure": 1013.0,
      "indoor_temp": 19.5,
      "target_temp": 20.0,
      "hour_of_day": 12,
      "day_of_week": 0,
      "is_weekend": 0,
      "is_night": 0,
      "outdoor_temp_avg_1h": 5.0,
      "outdoor_temp_avg_6h": 4.5,
      "outdoor_temp_avg_24h": 4.0,
      "indoor_temp_avg_6h": 19.2,
      "indoor_temp_avg_24h": 19.0,
      "target_temp_avg_6h": 20.0,
      "target_temp_avg_24h": 19.5,
      "heating_degree_hours_24h": 360.0,
      "heating_kwh_last_6h": 3.5,
      "heating_kwh_last_24h": 12.0
    }
  ],
  "update_historical": false
}
```

**Response:**

```json
{
  "status": "success",
  "predictions": [1.2345, 0.9876],
  "total_kwh": 2.2221,
  "model_info": {
    "features": ["outdoor_temp", "wind", ...],
    "training_timestamp": "2024-01-15T10:30:00"
  },
  "timeslots": ["2024-01-15T12:00:00", "2024-01-15T13:00:00"]
}
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `timeslots` | array | Optional timestamps for the predictions |
| `scenario_features` | array | Required. List of feature dictionaries (one per slot) |
| `update_historical` | boolean | If true, update historical kWh features based on prior predictions in the scenario |

> **Important:** Predictions should always start at the next or coming hour. This ensures accurate historical feature computation based on the latest available data.

---

### Predict Heating Demand - Simplified Scenario API ⭐ NEW

Predict heating demand using simplified, human-readable inputs. The system automatically calculates all required model features internally.

```http
POST /api/predictions/scenario
```

**Request body:**

```json
{
  "timeslots": [
    {
      "timestamp": "2024-01-15T14:00:00",
      "outdoor_temperature": 5.0,
      "wind_speed": 3.0,
      "humidity": 75.0,
      "pressure": 1013.0,
      "target_temperature": 20.0,
      "indoor_temperature": 19.5
    },
    {
      "timestamp": "2024-01-15T15:00:00",
      "outdoor_temperature": 4.5,
      "wind_speed": 3.5,
      "humidity": 76.0,
      "pressure": 1013.0,
      "target_temperature": 20.0
    }
  ]
}
```

**Response:**

```json
{
  "status": "success",
  "predictions": [
    {
      "timestamp": "2024-01-15T14:00:00",
      "predicted_kwh": 1.2345
    },
    {
      "timestamp": "2024-01-15T15:00:00",
      "predicted_kwh": 0.9876
    }
  ],
  "total_kwh": 2.2221,
  "slots_count": 2,
  "model_info": {
    "training_timestamp": "2024-01-15T10:30:00"
  }
}
```

**Required Fields (per timeslot):**

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | ISO 8601 datetime, **must be in the future** |
| `outdoor_temperature` | number | Forecasted outdoor temperature (°C) |
| `wind_speed` | number | Forecasted wind speed (m/s) |
| `humidity` | number | Forecasted relative humidity (%) |
| `pressure` | number | Forecasted air pressure (hPa) |
| `target_temperature` | number | Planned indoor setpoint (°C) |

**Optional Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `indoor_temperature` | number | Expected indoor temperature (°C). If not provided, uses target_temperature. |

**Automatically Computed by the System:**

The following features are derived internally and **should NOT be sent**:

- Time features: `hour_of_day`, `day_of_week`, `is_weekend`, `is_night`
- Historical aggregations: `outdoor_temp_avg_*`, `indoor_temp_avg_*`, `target_temp_avg_*`
- Heating metrics: `heating_degree_hours_24h`, `heating_kwh_last_*`

**Validation Rules:**

- All timestamps must be in the future; past timestamps are rejected
- All required fields must be present and non-null
- All numeric fields must be valid numbers

---

### Get Simplified Scenario Example

Get a pre-filled example for the simplified scenario API.

```http
GET /api/examples/scenario
```

**Response:**

```json
{
  "status": "success",
  "example": {
    "timeslots": [
      {
        "timestamp": "2024-01-15T14:00:00",
        "outdoor_temperature": 5.0,
        "wind_speed": 3.5,
        "humidity": 75.0,
        "pressure": 1013.0,
        "target_temperature": 20.0
      }
    ]
  },
  "description": "24-hour prediction with typical winter day temperature variation and setpoint schedule",
  "model_available": true,
  "required_fields": ["timestamp", "outdoor_temperature", "wind_speed", "humidity", "pressure", "target_temperature"],
  "optional_fields": ["indoor_temperature"]
}
```

---

### Enrich Scenario with Historical Features

Compute historical aggregation features from user-provided scenario data. This is useful when preparing prediction requests with user-specified weather forecasts.

```http
POST /api/predictions/enrich_scenario
```

**Request body:**

```json
{
  "scenario_features": [
    {"outdoor_temp": 5.0, "indoor_temp": 20.0, "target_temp": 21.0, "wind": 3.0, "humidity": 75.0, "pressure": 1013.0},
    {"outdoor_temp": 4.5, "indoor_temp": 20.0, "target_temp": 21.0, "wind": 3.5, "humidity": 76.0, "pressure": 1013.0}
  ],
  "timeslots": ["2024-01-15T12:00:00", "2024-01-15T13:00:00"]
}
```

**Response:**

```json
{
  "status": "success",
  "enriched_features": [
    {
      "outdoor_temp": 5.0,
      "outdoor_temp_avg_1h": 5.0,
      "outdoor_temp_avg_6h": 5.0,
      "outdoor_temp_avg_24h": 5.0,
      "heating_degree_hours_24h": 16.0,
      "hour_of_day": 12,
      "day_of_week": 0,
      "is_weekend": 0,
      "is_night": 0
    },
    ...
  ],
  "features_added": ["outdoor_temp_avg_1h", "outdoor_temp_avg_6h", ...]
}
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `scenario_features` | array | Required. List of feature dictionaries with base sensor values |
| `timeslots` | array | Optional. Timestamps for computing time features |

---

### Compare Predictions with Actual Data

Compare model predictions against actual historical data. This endpoint uses 5-minute average records to show the delta between model predictions and actual recorded values.

```http
POST /api/predictions/compare_actual
```

**Request body:**

```json
{
  "start_time": "2024-01-15T12:00:00",
  "end_time": "2024-01-15T18:00:00",
  "slot_duration_minutes": 60
}
```

**Response:**

```json
{
  "status": "success",
  "comparison": [
    {
      "slot_start": "2024-01-15T12:00:00",
      "actual_kwh": 1.25,
      "predicted_kwh": 1.18,
      "delta_kwh": -0.07,
      "delta_pct": -5.6
    },
    ...
  ],
  "summary": {
    "total_actual_kwh": 7.5,
    "total_predicted_kwh": 7.2,
    "mae_kwh": 0.08,
    "mape_pct": 6.5,
    "slots_compared": 6
  }
}
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_time` | string | Required. Start of the comparison period (ISO 8601) |
| `end_time` | string | Required. End of the comparison period (ISO 8601) |
| `slot_duration_minutes` | integer | Optional. Duration per slot (default 60 minutes) |

---

### Validate Prediction Start Time

Validate that a prediction start time is at the next hour or later.

```http
POST /api/predictions/validate_start_time
```

**Request body:**

```json
{
  "start_time": "2024-01-15T14:00:00"
}
```

**Response:**

```json
{
  "status": "success",
  "valid": true,
  "message": "Valid prediction start time: 2024-01-15 14:00:00",
  "next_valid_hour": "2024-01-15T14:00:00"
}
```

---

### Get Feature Configuration

Get the current feature configuration, including all core and experimental features.

```http
GET /api/features/config
```

**Response:**

```json
{
  "status": "success",
  "config": {
    "timezone": "Europe/Amsterdam",
    "core_feature_count": 13,
    "active_feature_count": 13,
    "experimental_enabled": {
      "pressure": false,
      "day_of_week": false,
      ...
    }
  },
  "features": {
    "weather": [...],
    "indoor": [...],
    "control": [...],
    "usage": [...],
    "time": [...]
  },
  "active_feature_names": ["outdoor_temp", "wind", "humidity", ...]
}
```

---

### Toggle Experimental Feature

Enable or disable an experimental feature. Core features cannot be toggled.

```http
POST /api/features/toggle
```

**Request body:**

```json
{
  "feature_name": "pressure",
  "enabled": true
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Feature 'pressure' is now enabled",
  "active_features": ["outdoor_temp", "wind", "humidity", "pressure", ...]
}
```

---

### Set Timezone

Set the timezone for time-based features (hour_of_day).

```http
POST /api/features/timezone
```

**Request body:**

```json
{
  "timezone": "Europe/Amsterdam"
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Timezone set to Europe/Amsterdam",
  "timezone": "Europe/Amsterdam"
}
```

---

### Get Feature Metadata

Get metadata for all features. This is the single source of truth for feature documentation.

```http
GET /api/features/metadata
```

**Response:**

```json
{
  "status": "success",
  "features": [
    {
      "name": "outdoor_temp",
      "category": "weather",
      "description": "Latest 5-minute outdoor temperature",
      "unit": "°C",
      "time_window": "none",
      "is_core": true,
      "enabled": true
    },
    ...
  ],
  "core_count": 13
}
```

---

## Usage Examples

### Basic Workflow

1. **Start the add-on** - Data collection begins automatically
2. **Wait for data** - Let the add-on collect at least 1-2 days of data
3. **Trigger resampling** - Call `POST /resample` via the UI button
4. **Train the model** - Call `POST /api/train/heating_demand`
5. **Make predictions** - Call `POST /api/predictions/scenario` (simplified) or `POST /api/predictions/heating_demand_profile` (advanced)

### Python Example: Simplified Scenario API (Recommended)

The simplified API is the easiest way to get predictions. Just provide weather forecasts and setpoint schedules - the system handles all feature engineering internally.

```python
import requests
from datetime import datetime, timedelta

# Check model status
status = requests.get("http://addon-url:8099/api/model/status")
print(status.json())

if status.json()["status"] == "available":
    # Calculate future timestamps
    now = datetime.now()
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    
    # Build simple scenario with weather forecast and setpoints
    scenario = {
        "timeslots": [
            {
                "timestamp": (next_hour).isoformat(),
                "outdoor_temperature": 5.0,      # Weather forecast
                "wind_speed": 3.0,
                "humidity": 75.0,
                "pressure": 1013.0,
                "target_temperature": 20.0,      # Your desired setpoint
            },
            {
                "timestamp": (next_hour + timedelta(hours=1)).isoformat(),
                "outdoor_temperature": 4.5,
                "wind_speed": 3.5,
                "humidity": 76.0,
                "pressure": 1013.0,
                "target_temperature": 20.0,
            },
        ]
    }
    
    response = requests.post(
        "http://addon-url:8099/api/predictions/scenario",
        json=scenario
    )
    
    result = response.json()
    print(f"Total predicted heating: {result['total_kwh']:.2f} kWh")
    for pred in result['predictions']:
        print(f"  {pred['timestamp']}: {pred['predicted_kwh']:.2f} kWh")
```

### Python Example: Advanced API (Full Control)

If you need full control over all model features, use the advanced API:

```python
import requests

# Check model status
status = requests.get("http://addon-url:8099/api/model/status")
print(status.json())

# If model is available, make a prediction
if status.json()["status"] == "available":
    features = status.json()["features"]
    
    # Build feature dictionary for next hour
    scenario = {
        "outdoor_temp": 5.0,
        "wind": 3.0,
        "humidity": 80.0,
        "pressure": 1013.0,
        "indoor_temp": 19.5,
        "target_temp": 20.0,
        "hour_of_day": 14,
        "day_of_week": 1,
        "is_weekend": 0,
        "is_night": 0,
        "outdoor_temp_avg_1h": 5.0,
        "outdoor_temp_avg_6h": 4.5,
        "outdoor_temp_avg_24h": 4.0,
        "indoor_temp_avg_6h": 19.2,
        "indoor_temp_avg_24h": 19.0,
        "target_temp_avg_6h": 20.0,
        "target_temp_avg_24h": 19.5,
        "heating_degree_hours_24h": 360.0,
        "heating_kwh_last_6h": 3.5,
        "heating_kwh_last_24h": 12.0
    }
    
    response = requests.post(
        "http://addon-url:8099/api/predictions/heating_demand_profile",
        json={"scenario_features": [scenario]}
    )
    
    result = response.json()
    print(f"Predicted heating demand: {result['predictions'][0]:.2f} kWh")
```

### Home Assistant Automation Example (Simplified API)

> **Note:** Replace `homeassistant.local:8099` with your actual add-on URL. When using ingress, you may need to use the internal add-on hostname.

```yaml
automation:
  - alias: "Get heating demand prediction (Simplified API)"
    trigger:
      - platform: time_pattern
        hours: "*"
        minutes: "0"
    action:
      - service: rest_command.get_heating_prediction_simple
        response_variable: prediction
      - service: input_number.set_value
        target:
          entity_id: input_number.predicted_heating_kwh
        data:
          value: "{{ prediction.content.predictions[0].predicted_kwh }}"

rest_command:
  get_heating_prediction_simple:
    url: "http://homeassistant.local:8099/api/predictions/scenario"
    method: POST
    content_type: application/json
    payload: >
      {
        "timeslots": [{
          "timestamp": "{{ (now() + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0).isoformat() }}",
          "outdoor_temperature": {{ states('sensor.outdoor_temperature') | float(5) }},
          "wind_speed": {{ states('sensor.wind_speed') | float(3) }},
          "humidity": {{ states('sensor.humidity') | float(75) }},
          "pressure": {{ states('sensor.air_pressure') | float(1013) }},
          "target_temperature": {{ states('sensor.thermostat_setpoint') | float(20) }}
        }]
      }
```

### Home Assistant Automation Example (Advanced API)

```yaml
automation:
  - alias: "Get heating demand prediction"
    trigger:
      - platform: time_pattern
        hours: "*"
        minutes: "0"
    action:
      - service: rest_command.get_heating_prediction
        response_variable: prediction
      - service: input_number.set_value
        target:
          entity_id: input_number.predicted_heating_kwh
        data:
          value: "{{ prediction.content.predictions[0] }}"

rest_command:
  get_heating_prediction:
    url: "http://homeassistant.local:8099/api/predictions/heating_demand_profile"
    method: POST
    content_type: application/json
    # Note: You need to provide ALL required features.
    # Check GET /api/model/status for the complete list of features.
    # The features below are examples - adapt to your actual sensor entity IDs.
    payload: >
      {
        "scenario_features": [{
          "outdoor_temp": {{ states('sensor.outdoor_temperature') | float(0) }},
          "wind": {{ states('sensor.wind_speed') | float(0) }},
          "humidity": {{ states('sensor.humidity') | float(80) }},
          "pressure": {{ states('sensor.air_pressure') | float(1013) }},
          "indoor_temp": {{ states('sensor.living_room_temperature') | float(20) }},
          "target_temp": {{ states('sensor.thermostat_setpoint') | float(20) }},
          "hour_of_day": {{ now().hour }},
          "day_of_week": {{ now().weekday() }},
          "is_weekend": {{ 1 if now().weekday() >= 5 else 0 }},
          "is_night": {{ 1 if now().hour >= 23 or now().hour < 7 else 0 }},
          "outdoor_temp_avg_1h": {{ states('sensor.outdoor_temperature') | float(0) }},
          "outdoor_temp_avg_6h": {{ states('sensor.outdoor_temperature') | float(0) }},
          "outdoor_temp_avg_24h": {{ states('sensor.outdoor_temperature') | float(0) }},
          "indoor_temp_avg_6h": {{ states('sensor.living_room_temperature') | float(20) }},
          "indoor_temp_avg_24h": {{ states('sensor.living_room_temperature') | float(20) }},
          "target_temp_avg_6h": {{ states('sensor.thermostat_setpoint') | float(20) }},
          "target_temp_avg_24h": {{ states('sensor.thermostat_setpoint') | float(20) }},
          "heating_degree_hours_24h": 300.0,
          "heating_kwh_last_6h": 2.5,
          "heating_kwh_last_24h": 10.0
        }]
      }
```

---

## Technical Details

### Database Schema

**samples** - Raw sensor readings
- `id` (INT, PK)
- `entity_id` (VARCHAR(128), indexed)
- `timestamp` (DATETIME, indexed)
- `value` (FLOAT)
- `unit` (VARCHAR(32), nullable)
- Unique constraint: (entity_id, timestamp)

**resampled_samples** - 5-minute aggregated data
- `id` (INT, PK)
- `slot_start` (DATETIME, indexed)
- `category` (VARCHAR(64), indexed)
- `value` (DOUBLE)
- `unit` (VARCHAR(32), nullable)
- Unique constraint: (slot_start, category)

**sensor_mappings** - Entity to category mappings
- `id` (INT, PK)
- `category` (VARCHAR(64), indexed)
- `entity_id` (VARCHAR(128), indexed)
- `is_active` (BOOLEAN)
- `priority` (INT)
- Unique constraint: (category, entity_id)

**sync_status** - Synchronization state tracking
- `entity_id` (VARCHAR(128), PK)
- `last_attempt` (DATETIME, nullable)
- `last_success` (DATETIME, nullable)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Home Assistant                            │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │ Temperature │   │   Weather   │   │ Heat Pump   │        │
│  │   Sensors   │   │   Sensors   │   │   Sensors   │        │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘        │
│         │                 │                 │                │
│         └────────────────┬┴─────────────────┘                │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                       │
│              │   History API         │                       │
│              └───────────┬───────────┘                       │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  Energy Orchestrator Add-on                   │
│                                                               │
│  ┌────────────────┐                                          │
│  │ Sensor Worker  │──────► samples table                     │
│  └────────────────┘             │                            │
│                                 ▼                            │
│  ┌────────────────┐     ┌──────────────┐                     │
│  │  Resample API  │────►│ resampled_   │                     │
│  └────────────────┘     │ samples      │                     │
│                         └──────┬───────┘                     │
│                                │                             │
│                                ▼                             │
│  ┌────────────────┐     ┌──────────────┐                     │
│  │  Training API  │────►│ Feature      │                     │
│  └────────────────┘     │ Engineering  │                     │
│                         └──────┬───────┘                     │
│                                │                             │
│                                ▼                             │
│  ┌────────────────┐     ┌──────────────┐     ┌────────────┐ │
│  │ Prediction API │◄────│    Model     │◄────│   /data/   │ │
│  └────────────────┘     │              │     │   *.joblib │ │
│                         └──────────────┘     └────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Error Handling

- **ModelNotAvailableError**: Raised when attempting predictions without a trained model
- **Database errors**: Logged and handled gracefully, sync continues
- **Missing features**: API returns 400 with list of missing features
- **Insufficient data**: Training returns 400 with data statistics

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

## License

This add-on is part of the [Kevin's Home Assistant Add-ons](/) repository.
