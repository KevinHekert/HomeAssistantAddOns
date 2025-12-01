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

#### Input Feature Categories

| Category | Description | Used As |
|----------|-------------|---------|
| `outdoor_temp` | Outdoor temperature | Input feature |
| `wind` | Wind speed | Input feature |
| `humidity` | Relative humidity | Input feature |
| `pressure` | Barometric pressure | Input feature |
| `indoor_temp` | Indoor temperature | Input feature |
| `target_temp` | Thermostat setpoint | Input feature |
| `hp_kwh_total` | Heat pump kWh counter | Target computation only |
| `dhw_active` | Hot water active flag | Filtering only |

#### Historical Aggregation Features

The model computes rolling statistics to capture thermal dynamics:

| Feature | Window | Description |
|---------|--------|-------------|
| `outdoor_temp_avg_1h` | 1 hour | Average outdoor temp (last 12 slots) |
| `outdoor_temp_avg_6h` | 6 hours | Average outdoor temp (last 72 slots) |
| `outdoor_temp_avg_24h` | 24 hours | Average outdoor temp (last 288 slots) |
| `outdoor_temp_avg_7d` | 7 days | Average outdoor temp (last 2016 slots)* |
| `indoor_temp_avg_6h` | 6 hours | Average indoor temp |
| `indoor_temp_avg_24h` | 24 hours | Average indoor temp |
| `target_temp_avg_6h` | 6 hours | Average setpoint |
| `target_temp_avg_24h` | 24 hours | Average setpoint |
| `heating_degree_hours_24h` | 24 hours | Sum of (target - outdoor)⁺ × (5/60) |
| `heating_degree_hours_7d` | 7 days | Sum of heating degree hours* |
| `heating_kwh_last_6h` | 6 hours | Sum of kWh consumption |
| `heating_kwh_last_24h` | 24 hours | Sum of kWh consumption |
| `heating_kwh_last_7d` | 7 days | Sum of kWh consumption* |

\* 7-day features are only included when at least 7 days (168 hours / 2016 five-minute slots) of history is available.

#### Time Features

| Feature | Description |
|---------|-------------|
| `hour_of_day` | Hour (0-23) |
| `day_of_week` | Day (0=Monday, 6=Sunday) |
| `is_weekend` | 1 if Saturday/Sunday, else 0 |
| `is_night` | 1 if hour is 23:00-06:59, else 0 |

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
    "has_7d_features": true
  }
}
```

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

---

## Usage Examples

### Basic Workflow

1. **Start the add-on** - Data collection begins automatically
2. **Wait for data** - Let the add-on collect at least 1-2 days of data
3. **Trigger resampling** - Call `POST /resample` via the UI button
4. **Train the model** - Call `POST /api/train/heating_demand`
5. **Make predictions** - Call `POST /api/predictions/heating_demand_profile`

### Python Example: Making Predictions

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

### Home Assistant Automation Example

> **Note:** Replace `homeassistant.local:8099` with your actual add-on URL. When using ingress, you may need to use the internal add-on hostname.

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
