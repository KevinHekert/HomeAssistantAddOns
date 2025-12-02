import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
import pandas as pd
from ha.ha_api import get_entity_state
from workers import start_sensor_logging_worker
from db.resample import resample_all_categories, get_sample_rate_minutes, set_sample_rate_minutes, VALID_SAMPLE_RATES, flush_resampled_samples
from db.core import init_db_schema
from db.sensor_config import sync_sensor_mappings
from db.samples import get_sensor_info
from db.sync_config import (
    get_sync_config,
    set_sync_config,
    MIN_BACKFILL_DAYS,
    MAX_BACKFILL_DAYS,
    MIN_SYNC_WINDOW_DAYS,
    MAX_SYNC_WINDOW_DAYS,
    MIN_SYNC_INTERVAL,
    MAX_SYNC_INTERVAL,
)
from ml.heating_features import (
    build_heating_feature_dataset,
    compute_scenario_historical_features,
    get_actual_vs_predicted_data,
    validate_prediction_start_time,
    validate_simplified_scenario,
    convert_simplified_to_model_features,
    get_available_historical_days,
    get_historical_day_hourly_data,
    SIMPLIFIED_REQUIRED_FIELDS,
    SIMPLIFIED_OPTIONAL_FIELDS,
    FeatureDatasetStats,
)
from ml.heating_demand_model import (
    HeatingDemandModel,
    ModelNotAvailableError,
    load_heating_demand_model,
    train_heating_demand_model,
    predict_scenario,
)
from ml.two_step_model import (
    TwoStepHeatingDemandModel,
    TwoStepModelNotAvailableError,
    load_two_step_heating_demand_model,
    train_two_step_heating_demand_model,
    predict_two_step_scenario,
)
from ml.feature_config import (
    get_feature_config,
    reload_feature_config,
    get_feature_metadata_dict,
    get_core_feature_count,
    FeatureCategory,
)


app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
_Logger = logging.getLogger(__name__)

# Voor nu één sensor; later lijst vanuit config/integratie
WIND_ENTITY_ID = "sensor.knmi_windsnelheid"

# Precision for prediction output values
PREDICTION_DECIMAL_PLACES = 4

# Global model instances
_heating_model: HeatingDemandModel | None = None
_two_step_model: TwoStepHeatingDemandModel | None = None


def _get_model() -> HeatingDemandModel | None:
    """Get the global heating demand model, loading it if necessary."""
    global _heating_model
    if _heating_model is None:
        try:
            _heating_model = load_heating_demand_model()
        except Exception as e:
            _Logger.error("Failed to load heating demand model: %s", e)
    return _heating_model


def _get_two_step_model() -> TwoStepHeatingDemandModel | None:
    """Get the global two-step heating demand model, loading it if necessary."""
    global _two_step_model
    if _two_step_model is None:
        try:
            _two_step_model = load_two_step_heating_demand_model()
        except Exception as e:
            _Logger.error("Failed to load two-step heating demand model: %s", e)
    return _two_step_model


@app.get("/")
def index():
    # Alleen actuele waarde ophalen voor de UI
    wind_speed, wind_unit = get_entity_state(WIND_ENTITY_ID)

    return render_template(
        "index.html",
        wind_speed=wind_speed,
        wind_unit=wind_unit,
    )


@app.post("/resample")
def trigger_resample():
    """Trigger resampling of all categories to configured time slots.
    
    Optionally accepts a JSON body with:
    {
        "sample_rate_minutes": 5,  // Optional, overrides configured rate
        "flush": true              // Optional, flush existing data before resampling
    }
    
    Valid sample rates are: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60
    These are divisors of 60 that ensure proper hour boundary alignment.
    
    When the sample rate changes, the flush parameter should be set to true
    to clear existing resampled data that was computed with a different interval.
    """
    try:
        # Check if parameters were provided in the request
        sample_rate = None
        flush = False
        if request.is_json:
            data = request.get_json()
            if data:
                if "sample_rate_minutes" in data:
                    sample_rate = int(data["sample_rate_minutes"])
                    if sample_rate not in VALID_SAMPLE_RATES:
                        return jsonify({
                            "status": "error",
                            "message": f"sample_rate_minutes must be one of {VALID_SAMPLE_RATES}",
                        }), 400
                if "flush" in data:
                    flush = bool(data["flush"])
        
        _Logger.info("Resample triggered via UI with sample_rate=%s, flush=%s", sample_rate or "default", flush)
        
        # Always use resample_all_categories - it uses configured rate when sample_rate is None
        stats = resample_all_categories(sample_rate, flush=flush)
        
        return jsonify({
            "status": "success",
            "message": "Resampling completed successfully",
            "stats": {
                "slots_processed": stats.slots_processed,
                "slots_saved": stats.slots_saved,
                "slots_skipped": stats.slots_skipped,
                "categories": stats.categories,
                "start_time": stats.start_time.isoformat() if stats.start_time else None,
                "end_time": stats.end_time.isoformat() if stats.end_time else None,
                "sample_rate_minutes": stats.sample_rate_minutes,
                "table_flushed": stats.table_flushed,
            },
        })
    except Exception as e:
        _Logger.error("Error during resampling: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/sample_rate")
def get_sample_rate():
    """Get the current sample rate configuration."""
    try:
        rate = get_sample_rate_minutes()
        return jsonify({
            "status": "success",
            "sample_rate_minutes": rate,
            "valid_rates": VALID_SAMPLE_RATES,
            "description": f"Data is sampled into {rate}-minute time slots",
        })
    except Exception as e:
        _Logger.error("Error getting sample rate: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/sample_rate")
def update_sample_rate():
    """Update the sample rate configuration.
    
    Request body:
    {
        "sample_rate_minutes": 5  // Must be one of VALID_SAMPLE_RATES
    }
    
    Response:
    {
        "status": "success",
        "sample_rate_minutes": 5,
        "message": "Sample rate updated to 5 minutes",
        "note": "Flush existing resampled data and resample to use the new rate"
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Request body required",
                "valid_rates": VALID_SAMPLE_RATES,
            }), 400
        
        data = request.get_json()
        if not data or "sample_rate_minutes" not in data:
            return jsonify({
                "status": "error",
                "message": "sample_rate_minutes is required",
                "valid_rates": VALID_SAMPLE_RATES,
            }), 400
        
        try:
            rate = int(data["sample_rate_minutes"])
        except (TypeError, ValueError):
            return jsonify({
                "status": "error",
                "message": "sample_rate_minutes must be an integer",
                "valid_rates": VALID_SAMPLE_RATES,
            }), 400
        
        if rate not in VALID_SAMPLE_RATES:
            return jsonify({
                "status": "error",
                "message": f"sample_rate_minutes must be one of {VALID_SAMPLE_RATES}",
                "valid_rates": VALID_SAMPLE_RATES,
            }), 400
        
        if set_sample_rate_minutes(rate):
            _Logger.info("Sample rate updated to %d minutes via API", rate)
            return jsonify({
                "status": "success",
                "sample_rate_minutes": rate,
                "message": f"Sample rate updated to {rate} minutes",
                "note": "Flush existing resampled data and resample to use the new rate",
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to save sample rate configuration",
            }), 500
    except Exception as e:
        _Logger.error("Error updating sample rate: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/sync_config")
def get_sync_config_api():
    """Get the current sync configuration.
    
    Response:
    {
        "status": "success",
        "config": {
            "backfill_days": 14,
            "sync_window_days": 1,
            "sensor_sync_interval": 1,
            "sensor_loop_interval": 1
        },
        "limits": {
            "backfill_days": {"min": 1, "max": 365},
            "sync_window_days": {"min": 1, "max": 30},
            "sensor_sync_interval": {"min": 1, "max": 3600},
            "sensor_loop_interval": {"min": 1, "max": 3600}
        }
    }
    """
    try:
        config = get_sync_config()
        return jsonify({
            "status": "success",
            "config": {
                "backfill_days": config.backfill_days,
                "sync_window_days": config.sync_window_days,
                "sensor_sync_interval": config.sensor_sync_interval,
                "sensor_loop_interval": config.sensor_loop_interval,
            },
            "limits": {
                "backfill_days": {"min": MIN_BACKFILL_DAYS, "max": MAX_BACKFILL_DAYS},
                "sync_window_days": {"min": MIN_SYNC_WINDOW_DAYS, "max": MAX_SYNC_WINDOW_DAYS},
                "sensor_sync_interval": {"min": MIN_SYNC_INTERVAL, "max": MAX_SYNC_INTERVAL},
                "sensor_loop_interval": {"min": MIN_SYNC_INTERVAL, "max": MAX_SYNC_INTERVAL},
            },
        })
    except Exception as e:
        _Logger.error("Error getting sync config: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/sync_config")
def update_sync_config():
    """Update the sync configuration.
    
    Request body (all fields optional, only provided fields are updated):
    {
        "backfill_days": 14,          // Days to backfill when no samples exist (1-365)
        "sync_window_days": 1,        // Size of each sync window in days (1-30)
        "sensor_sync_interval": 1,    // Wait time in seconds between sensors (1-3600)
        "sensor_loop_interval": 1     // Wait time in seconds between sync loops (1-3600)
    }
    
    Response:
    {
        "status": "success",
        "message": "Sync configuration updated",
        "config": {...}
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "At least one configuration field is required",
            }), 400
        
        # Parse values
        backfill_days = None
        sync_window_days = None
        sensor_sync_interval = None
        sensor_loop_interval = None
        
        if "backfill_days" in data:
            try:
                backfill_days = int(data["backfill_days"])
            except (TypeError, ValueError):
                return jsonify({
                    "status": "error",
                    "message": "backfill_days must be an integer",
                }), 400
        
        if "sync_window_days" in data:
            try:
                sync_window_days = int(data["sync_window_days"])
            except (TypeError, ValueError):
                return jsonify({
                    "status": "error",
                    "message": "sync_window_days must be an integer",
                }), 400
        
        if "sensor_sync_interval" in data:
            try:
                sensor_sync_interval = int(data["sensor_sync_interval"])
            except (TypeError, ValueError):
                return jsonify({
                    "status": "error",
                    "message": "sensor_sync_interval must be an integer",
                }), 400
        
        if "sensor_loop_interval" in data:
            try:
                sensor_loop_interval = int(data["sensor_loop_interval"])
            except (TypeError, ValueError):
                return jsonify({
                    "status": "error",
                    "message": "sensor_loop_interval must be an integer",
                }), 400
        
        # Update configuration
        success, error = set_sync_config(
            backfill_days=backfill_days,
            sync_window_days=sync_window_days,
            sensor_sync_interval=sensor_sync_interval,
            sensor_loop_interval=sensor_loop_interval,
        )
        
        if not success:
            return jsonify({
                "status": "error",
                "message": error or "Failed to save configuration",
            }), 400
        
        # Return updated config
        config = get_sync_config()
        _Logger.info("Sync config updated via API: %s", config)
        
        return jsonify({
            "status": "success",
            "message": "Sync configuration updated",
            "config": {
                "backfill_days": config.backfill_days,
                "sync_window_days": config.sync_window_days,
                "sensor_sync_interval": config.sensor_sync_interval,
                "sensor_loop_interval": config.sensor_loop_interval,
            },
        })
        
    except Exception as e:
        _Logger.error("Error updating sync config: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# Sensor units mapping for display in UI
SENSOR_UNITS = {
    "outdoor_temp": "°C",
    "indoor_temp": "°C",
    "target_temp": "°C",
    "dhw_temp": "°C",
    "flow_temp": "°C",
    "return_temp": "°C",
    "wind": "m/s",
    "humidity": "%",
    "pressure": "hPa",
    "hp_kwh_total": "kWh",
    "dhw_active": "",
}


def _build_training_data_response(stats: FeatureDatasetStats) -> dict:
    """
    Build the training_data response with all sensor categories.
    
    For hp_kwh_total, shows the delta (energy consumed) instead of raw cumulative values.
    """
    training_data = {}
    
    for category, range_data in stats.sensor_ranges.items():
        if category == "hp_kwh_total":
            # Show delta instead of first/last cumulative values
            training_data[category] = {
                "delta": stats.hp_kwh_delta,
                "first": range_data.first,
                "last": range_data.last,
                "unit": SENSOR_UNITS.get(category, ""),
            }
        else:
            training_data[category] = {
                "first": range_data.first,
                "last": range_data.last,
                "unit": SENSOR_UNITS.get(category, ""),
            }
    
    return training_data


@app.post("/api/train/heating_demand")
def train_heating_demand():
    """Train the heating demand prediction model."""
    global _heating_model
    try:
        _Logger.info("Training heating demand model...")
        
        # Build feature dataset
        df, stats = build_heating_feature_dataset(min_samples=50)
        
        if df is None:
            return jsonify({
                "status": "error",
                "message": "Insufficient data for training",
                "stats": {
                    "total_slots": stats.total_slots,
                    "valid_slots": stats.valid_slots,
                    "dropped_missing_features": stats.dropped_missing_features,
                    "dropped_missing_target": stats.dropped_missing_target,
                },
            }), 400
        
        # Train model
        model, metrics = train_heating_demand_model(df)
        _heating_model = model
        
        return jsonify({
            "status": "success",
            "message": "Model trained successfully",
            "metrics": {
                "train_samples": metrics.train_samples,
                "val_samples": metrics.val_samples,
                "train_mae_kwh": round(metrics.train_mae, 4),
                "val_mae_kwh": round(metrics.val_mae, 4),
                "val_mape_pct": round(metrics.val_mape * 100, 2) if metrics.val_mape == metrics.val_mape else None,
                "val_r2": round(metrics.val_r2, 4),
                "features": metrics.features,
            },
            "dataset_stats": {
                "total_slots": stats.total_slots,
                "valid_slots": stats.valid_slots,
                "features_used": stats.features_used,
                "has_7d_features": stats.has_7d_features,
                "data_start_time": stats.data_start_time.isoformat() if stats.data_start_time else None,
                "data_end_time": stats.data_end_time.isoformat() if stats.data_end_time else None,
                "available_history_hours": round(stats.available_history_hours, 1) if stats.available_history_hours else None,
            },
            "training_data": _build_training_data_response(stats),
        })
        
    except Exception as e:
        _Logger.error("Error training heating demand model: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/predictions/heating_demand_profile")
def predict_heating_demand_profile():
    """
    Predict heating demand for a scenario.
    
    Request body:
    {
        "timeslots": ["2024-01-01T12:00:00", "2024-01-01T13:00:00", ...],
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
                ...
            },
            ...
        ],
        "update_historical": false
    }
    
    Response:
    {
        "predictions": [1.2, 0.8, ...],
        "total_kwh": 24.5,
        "model_info": {...}
    }
    """
    model = _get_model()
    
    if model is None or not model.is_available:
        return jsonify({
            "status": "error",
            "message": "Model not trained. Please train the model first via POST /api/train/heating_demand",
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        scenario_features = data.get("scenario_features", [])
        timeslots = data.get("timeslots", [])
        update_historical = data.get("update_historical", False)
        
        if not scenario_features:
            return jsonify({
                "status": "error",
                "message": "scenario_features is required and must be non-empty",
            }), 400
        
        # Validate features
        missing_features = []
        for i, features in enumerate(scenario_features):
            missing = [f for f in model.feature_names if f not in features]
            if missing:
                missing_features.append({"slot": i, "missing": missing})
        
        if missing_features:
            return jsonify({
                "status": "error",
                "message": "Missing required features",
                "details": missing_features,
                "required_features": model.feature_names,
            }), 400
        
        # Make predictions
        predictions = predict_scenario(
            model,
            scenario_features,
            update_historical=update_historical,
        )
        
        return jsonify({
            "status": "success",
            "predictions": [round(p, 4) for p in predictions],
            "total_kwh": round(sum(predictions), 4),
            "model_info": {
                "features": model.feature_names,
                "training_timestamp": model.training_timestamp.isoformat() if model.training_timestamp else None,
            },
            "timeslots": timeslots if timeslots else None,
        })
        
    except ModelNotAvailableError:
        return jsonify({
            "status": "error",
            "message": "Model not available",
        }), 503
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        _Logger.error("Error predicting heating demand: %s", e, exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@app.get("/api/model/status")
def get_model_status():
    """Get the status of the heating demand model."""
    model = _get_model()
    
    if model is None or not model.is_available:
        return jsonify({
            "status": "not_available",
            "message": "No trained model available",
        })
    
    return jsonify({
        "status": "available",
        "features": model.feature_names,
        "training_timestamp": model.training_timestamp.isoformat() if model.training_timestamp else None,
    })


@app.get("/api/sensors/info")
def get_sensors_info():
    """Get information about all sensors (first and last timestamp per sensor)."""
    try:
        sensors = get_sensor_info()
        return jsonify({
            "status": "success",
            "sensors": sensors,
            "count": len(sensors),
        })
    except Exception as e:
        _Logger.error("Error getting sensor info: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/predictions/enrich_scenario")
def enrich_scenario_with_historical_features():
    """
    Enrich user-provided scenario features with computed historical aggregations.
    
    This endpoint helps users prepare their prediction requests by computing
    historical aggregation features (1h, 6h, 24h averages) from the provided
    scenario data.
    
    Request body:
    {
        "scenario_features": [
            {"outdoor_temp": 5.0, "wind": 3.0, "humidity": 75.0, ...},
            {"outdoor_temp": 4.5, "wind": 3.5, "humidity": 76.0, ...},
            ...
        ],
        "timeslots": ["2024-01-15T12:00:00", "2024-01-15T13:00:00", ...]  // optional
    }
    
    Response:
    {
        "status": "success",
        "enriched_features": [
            {
                "outdoor_temp": 5.0,
                "outdoor_temp_avg_1h": 5.0,
                "outdoor_temp_avg_6h": 5.0,
                "heating_degree_hours_24h": 360.0,
                ...
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        scenario_features = data.get("scenario_features", [])
        timeslots_str = data.get("timeslots", [])
        
        if not scenario_features:
            return jsonify({
                "status": "error",
                "message": "scenario_features is required and must be non-empty",
            }), 400
        
        # Parse timeslots if provided
        timeslots = None
        if timeslots_str:
            try:
                timeslots = [
                    datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    for ts in timeslots_str
                ]
            except ValueError as e:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid timeslot format: {e}",
                }), 400
        
        # Compute historical features
        enriched = compute_scenario_historical_features(
            scenario_features,
            timeslots=timeslots,
        )
        
        return jsonify({
            "status": "success",
            "enriched_features": enriched,
            "features_added": [
                "outdoor_temp_avg_1h", "outdoor_temp_avg_6h", "outdoor_temp_avg_24h",
                "indoor_temp_avg_6h", "indoor_temp_avg_24h",
                "target_temp_avg_6h", "target_temp_avg_24h",
                "heating_degree_hours_24h",
                "hour_of_day", "day_of_week", "is_weekend", "is_night",
            ],
        })
        
    except Exception as e:
        _Logger.error("Error enriching scenario: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/predictions/compare_actual")
def compare_predictions_with_actual():
    """
    Compare model predictions with actual historical data.
    
    This endpoint allows users to validate model accuracy by comparing
    predictions against actual recorded data. It uses 5-minute average
    records to compute the actual values and shows the delta between
    the model's predictions and reality.
    
    Request body:
    {
        "start_time": "2024-01-15T12:00:00",
        "end_time": "2024-01-15T18:00:00",
        "slot_duration_minutes": 60  // optional, default 60
    }
    
    Response:
    {
        "status": "success",
        "comparison": [
            {
                "slot_start": "2024-01-15T12:00:00",
                "actual_kwh": 1.25,
                "predicted_kwh": 1.18,
                "delta_kwh": -0.07,
                "delta_pct": -5.6,
                "features": {...}
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
    """
    model = _get_model()
    
    if model is None or not model.is_available:
        return jsonify({
            "status": "error",
            "message": "Model not trained. Please train the model first via POST /api/train/heating_demand",
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        start_time_str = data.get("start_time")
        end_time_str = data.get("end_time")
        slot_duration = data.get("slot_duration_minutes", 60)
        
        if not start_time_str or not end_time_str:
            return jsonify({
                "status": "error",
                "message": "start_time and end_time are required",
            }), 400
        
        try:
            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": f"Invalid datetime format: {e}",
            }), 400
        
        # Get actual historical data
        actual_df, error = get_actual_vs_predicted_data(
            start_time,
            end_time,
            slot_duration_minutes=slot_duration,
        )
        
        if actual_df is None:
            return jsonify({
                "status": "error",
                "message": error or "No data available for the specified time range",
            }), 404
        
        # Prepare features for predictions
        comparison_results = []
        predictions = []
        actuals = []
        
        for _, row in actual_df.iterrows():
            # Build feature dictionary from actual data
            features = {}
            missing_features = []
            
            for feat in model.feature_names:
                if feat in row and row[feat] is not None and not pd.isna(row[feat]):
                    features[feat] = float(row[feat])
                else:
                    missing_features.append(feat)
            
            # Skip rows with missing features
            if missing_features:
                _Logger.debug("Skipping slot %s: missing features %s", row["slot_start"], missing_features)
                continue
            
            # Get actual value
            actual_kwh = row.get("actual_heating_kwh")
            if actual_kwh is None or pd.isna(actual_kwh):
                continue
            
            # Make prediction
            try:
                predicted_kwh = model.predict(features)
            except Exception as e:
                _Logger.warning("Prediction failed for slot %s: %s", row["slot_start"], e)
                continue
            
            delta_kwh = predicted_kwh - actual_kwh
            delta_pct = (delta_kwh / actual_kwh * 100) if actual_kwh > 0.01 else None
            
            comparison_results.append({
                "slot_start": row["slot_start"].isoformat() if hasattr(row["slot_start"], "isoformat") else str(row["slot_start"]),
                "actual_kwh": round(actual_kwh, 4),
                "predicted_kwh": round(predicted_kwh, 4),
                "delta_kwh": round(delta_kwh, 4),
                "delta_pct": round(delta_pct, 1) if delta_pct is not None else None,
            })
            
            predictions.append(predicted_kwh)
            actuals.append(actual_kwh)
        
        if not comparison_results:
            return jsonify({
                "status": "error",
                "message": "No valid slots found for comparison. Check that all required features are available.",
            }), 404
        
        # Compute summary statistics
        import numpy as np
        predictions_arr = np.array(predictions)
        actuals_arr = np.array(actuals)
        
        mae = np.mean(np.abs(predictions_arr - actuals_arr))
        
        # MAPE (exclude near-zero actuals)
        nonzero_mask = actuals_arr > 0.01
        if nonzero_mask.any():
            mape = np.mean(np.abs((predictions_arr[nonzero_mask] - actuals_arr[nonzero_mask]) / actuals_arr[nonzero_mask])) * 100
        else:
            mape = None
        
        return jsonify({
            "status": "success",
            "comparison": comparison_results,
            "summary": {
                "total_actual_kwh": round(sum(actuals), 4),
                "total_predicted_kwh": round(sum(predictions), 4),
                "mae_kwh": round(mae, 4),
                "mape_pct": round(mape, 1) if mape is not None else None,
                "slots_compared": len(comparison_results),
            },
        })
        
    except Exception as e:
        _Logger.error("Error comparing predictions: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/predictions/validate_start_time")
def validate_start_time():
    """
    Validate that a prediction start time is valid (next hour or later).
    
    Request body:
    {
        "start_time": "2024-01-15T14:00:00"
    }
    
    Response:
    {
        "status": "success",
        "valid": true,
        "message": "Valid prediction start time: 2024-01-15 14:00:00",
        "next_valid_hour": "2024-01-15T14:00:00"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        start_time_str = data.get("start_time")
        
        if not start_time_str:
            return jsonify({
                "status": "error",
                "message": "start_time is required",
            }), 400
        
        try:
            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": f"Invalid datetime format: {e}",
            }), 400
        
        is_valid, message = validate_prediction_start_time(start_time)
        
        # Compute the next valid hour for convenience
        from datetime import timedelta
        now = datetime.now()
        next_valid = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        return jsonify({
            "status": "success",
            "valid": is_valid,
            "message": message,
            "next_valid_hour": next_valid.isoformat(),
        })
        
    except Exception as e:
        _Logger.error("Error validating start time: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/predictions/scenario")
def predict_heating_demand_scenario():
    """
    Predict heating demand using simplified scenario inputs.
    
    This endpoint accepts human-readable inputs instead of low-level model features.
    All time-based features and historical aggregations are computed internally.
    
    When two-step prediction is enabled in the feature configuration and the two-step
    model is available, this endpoint will automatically use the two-step model for
    improved accuracy.
    
    Request body:
    {
        "timeslots": [
            {
                "timestamp": "2024-01-15T14:00:00",     // Required, ISO 8601, must be future
                "outdoor_temperature": 5.0,             // Required
                "wind_speed": 3.0,                      // Required
                "humidity": 75.0,                       // Required
                "pressure": 1013.0,                     // Required
                "target_temperature": 20.0,             // Required
                "indoor_temperature": 19.5              // Optional
            },
            ...
        ]
    }
    
    Response:
    {
        "status": "success",
        "predictions": [
            {
                "timestamp": "2024-01-15T14:00:00",
                "predicted_kwh": 1.2345
            },
            ...
        ],
        "total_kwh": 24.5,
        "model_info": {...}
    }
    
    Errors:
    - 400: Missing required fields, invalid timestamps, past timestamps
    - 503: Model not trained
    """
    # Check if two-step prediction is enabled and the model is available
    config = get_feature_config()
    use_two_step = False
    two_step_model = None
    
    if config.is_two_step_prediction_enabled():
        two_step_model = _get_two_step_model()
        if two_step_model is not None and two_step_model.is_available:
            use_two_step = True
    
    # Get the appropriate model
    if use_two_step:
        model = two_step_model
    else:
        model = _get_model()
    
    if model is None or not model.is_available:
        return jsonify({
            "status": "error",
            "message": "Model not trained. Please train the model first via POST /api/train/heating_demand",
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        timeslots = data.get("timeslots", [])
        
        if not timeslots:
            return jsonify({
                "status": "error",
                "message": "timeslots is required and must be non-empty",
                "required_fields": SIMPLIFIED_REQUIRED_FIELDS,
                "optional_fields": SIMPLIFIED_OPTIONAL_FIELDS,
            }), 400
        
        # Validate simplified scenario input
        validation = validate_simplified_scenario(timeslots)
        
        if not validation.valid:
            return jsonify({
                "status": "error",
                "message": "Validation failed",
                "errors": validation.errors,
                "required_fields": SIMPLIFIED_REQUIRED_FIELDS,
                "optional_fields": SIMPLIFIED_OPTIONAL_FIELDS,
            }), 400
        
        # Convert simplified input to model features
        model_features, parsed_timestamps = convert_simplified_to_model_features(
            timeslots,
            model.feature_names,
            include_historical_heating=True,
        )
        
        # Make predictions using the appropriate model
        if use_two_step:
            # Use two-step model
            two_step_predictions = predict_two_step_scenario(
                model,
                model_features,
                update_historical=True,
            )
            
            # Build response with two-step prediction details
            prediction_results = []
            total_kwh = 0.0
            active_count = 0
            inactive_count = 0
            
            for ts, pred in zip(parsed_timestamps, two_step_predictions):
                prediction_results.append({
                    "timestamp": ts.isoformat(),
                    "predicted_kwh": round(pred.predicted_kwh, PREDICTION_DECIMAL_PLACES),
                    "is_active": pred.is_active,
                    "activity_probability": round(pred.classifier_probability, 4),
                })
                total_kwh += pred.predicted_kwh
                if pred.is_active:
                    active_count += 1
                else:
                    inactive_count += 1
            
            return jsonify({
                "status": "success",
                "predictions": prediction_results,
                "total_kwh": round(total_kwh, PREDICTION_DECIMAL_PLACES),
                "slots_count": len(two_step_predictions),
                "model_info": {
                    "training_timestamp": model.training_timestamp.isoformat() if model.training_timestamp else None,
                    "two_step_prediction": True,
                    "activity_threshold_kwh": round(model.activity_threshold_kwh, 4),
                },
                "summary": {
                    "active_hours": active_count,
                    "inactive_hours": inactive_count,
                },
            })
        else:
            # Use standard single-step model
            predictions = predict_scenario(
                model,
                model_features,
                update_historical=True,
            )
            
            # Build response with timestamp-prediction pairs
            prediction_results = []
            for ts, pred in zip(parsed_timestamps, predictions):
                prediction_results.append({
                    "timestamp": ts.isoformat(),
                    "predicted_kwh": round(pred, PREDICTION_DECIMAL_PLACES),
                })
            
            return jsonify({
                "status": "success",
                "predictions": prediction_results,
                "total_kwh": round(sum(predictions), PREDICTION_DECIMAL_PLACES),
                "slots_count": len(predictions),
                "model_info": {
                    "training_timestamp": model.training_timestamp.isoformat() if model.training_timestamp else None,
                    "two_step_prediction": False,
                },
            })
        
    except ModelNotAvailableError:
        return jsonify({
            "status": "error",
            "message": "Model not available",
        }), 503
    except TwoStepModelNotAvailableError:
        return jsonify({
            "status": "error",
            "message": "Two-step model not available",
        }), 503
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        _Logger.error("Error predicting heating demand scenario: %s", e, exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@app.get("/api/examples/scenario")
def get_scenario_example():
    """
    Get a pre-filled example for the simplified scenario API.
    
    Returns an example request body with 24 hours of future predictions
    using the simplified input format.
    """
    from datetime import timedelta
    
    # Start at the next hour
    now = datetime.now()
    start_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    
    # Generate 24 hours of example timeslots
    example_timeslots = []
    for hour_offset in range(24):
        ts = start_hour + timedelta(hours=hour_offset)
        hour = ts.hour
        
        # Simulate realistic outdoor temperature variation
        if 0 <= hour < 6:
            outdoor_temp = 2.0
        elif 6 <= hour < 10:
            outdoor_temp = 3.0 + (hour - 6) * 0.5
        elif 10 <= hour < 14:
            outdoor_temp = 5.0 + (hour - 10) * 0.5
        elif 14 <= hour < 18:
            outdoor_temp = 7.0 - (hour - 14) * 0.5
        else:
            outdoor_temp = 5.0 - (hour - 18) * 0.5
        
        # Setpoint schedule
        if 6 <= hour < 22:
            target_temp = 20.0
        else:
            target_temp = 17.0
        
        example_timeslots.append({
            "timestamp": ts.isoformat(),
            "outdoor_temperature": round(outdoor_temp, 1),
            "wind_speed": 3.5,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": target_temp,
        })
    
    model = _get_model()
    
    return jsonify({
        "status": "success",
        "example": {
            "timeslots": example_timeslots,
        },
        "description": "24-hour prediction with typical winter day temperature variation and setpoint schedule",
        "model_available": model is not None and model.is_available,
        "required_fields": SIMPLIFIED_REQUIRED_FIELDS,
        "optional_fields": SIMPLIFIED_OPTIONAL_FIELDS,
    })


@app.get("/api/examples/available_days")
def get_available_days():
    """
    Get list of available historical days that can be used as scenario examples.
    
    Returns days from the 5-minute resampled data, excluding the first and last day
    to ensure complete data is available.
    
    Response:
    {
        "status": "success",
        "days": ["2024-01-02", "2024-01-03", ...],
        "count": 10
    }
    """
    try:
        days = get_available_historical_days()
        
        return jsonify({
            "status": "success",
            "days": days,
            "count": len(days),
        })
    except Exception as e:
        _Logger.error("Error getting available days: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/examples/historical_day/<date_str>")
def get_historical_day_example(date_str: str):
    """
    Get hourly averaged data for a specific historical day.
    
    This endpoint retrieves actual historical data from the 5-minute samples,
    aggregated to hourly averages, for use as a scenario example.
    
    Path parameter:
        date_str: Date in YYYY-MM-DD format
        
    Response:
    {
        "status": "success",
        "date": "2024-01-15",
        "hourly_data": [
            {
                "timestamp": "2024-01-15T00:00:00",
                "outdoor_temperature": 5.0,
                "wind_speed": 3.0,
                "humidity": 75.0,
                "pressure": 1013.0,
                "target_temperature": 17.0,
                "indoor_temperature": 19.5,
                "actual_heating_kwh": 1.25
            },
            ...
        ],
        "scenario_format": {...}  // Ready-to-use format for /api/predictions/scenario
    }
    """
    try:
        data, error = get_historical_day_hourly_data(date_str)
        
        if data is None:
            return jsonify({
                "status": "error",
                "message": error or "No data available",
            }), 404
        
        # Build scenario format (convert to future timestamps for prediction)
        # Timestamps are adjusted to be 2 days after today (day after tomorrow)
        # while keeping the same hour of day
        today = datetime.now().date()
        prediction_date = today + timedelta(days=2)
        
        scenario_timeslots = []
        for hour in data:
            # Parse the original timestamp to extract the hour
            original_ts = datetime.fromisoformat(hour["timestamp"])
            # Create new timestamp with prediction date but same hour
            prediction_ts = datetime(
                year=prediction_date.year,
                month=prediction_date.month,
                day=prediction_date.day,
                hour=original_ts.hour,
                minute=original_ts.minute,
                second=original_ts.second
            )
            slot = {
                "timestamp": prediction_ts.isoformat(),
            }
            # Add required fields
            if "outdoor_temperature" in hour:
                slot["outdoor_temperature"] = hour["outdoor_temperature"]
            if "wind_speed" in hour:
                slot["wind_speed"] = hour["wind_speed"]
            if "humidity" in hour:
                slot["humidity"] = hour["humidity"]
            if "pressure" in hour:
                slot["pressure"] = hour["pressure"]
            if "target_temperature" in hour:
                slot["target_temperature"] = hour["target_temperature"]
            # Add optional fields
            if "indoor_temperature" in hour:
                slot["indoor_temperature"] = hour["indoor_temperature"]
            
            scenario_timeslots.append(slot)
        
        model = _get_model()
        
        return jsonify({
            "status": "success",
            "date": date_str,
            "hourly_data": data,
            "scenario_format": {
                "timeslots": scenario_timeslots,
            },
            "model_available": model is not None and model.is_available,
            "description": f"Historical data for {date_str} with hourly averages",
        })
        
    except Exception as e:
        _Logger.error("Error getting historical day data: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/features/config")
def get_features_config():
    """
    Get the current feature configuration.
    
    Returns all features (core + experimental) with their metadata and current status.
    
    Response:
    {
        "status": "success",
        "config": {
            "timezone": "Europe/Amsterdam",
            "core_feature_count": 13,
            "active_feature_count": 13,
            "experimental_enabled": {"pressure": false, ...}
        },
        "features": {
            "weather": [...],
            "indoor": [...],
            "control": [...],
            "usage": [...],
            "time": [...]
        }
    }
    """
    try:
        config = get_feature_config()
        grouped = config.get_features_by_category()
        
        # Convert features to serializable format
        features_by_category = {}
        for category, features in grouped.items():
            features_by_category[category] = [f.to_dict() for f in features]
        
        return jsonify({
            "status": "success",
            "config": {
                "timezone": config.timezone,
                "core_feature_count": get_core_feature_count(),
                "active_feature_count": len(config.get_active_feature_names()),
                "experimental_enabled": config.experimental_enabled,
            },
            "features": features_by_category,
            "active_feature_names": config.get_active_feature_names(),
        })
    except Exception as e:
        _Logger.error("Error getting feature config: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/features/toggle")
def toggle_experimental_feature():
    """
    Enable or disable an experimental feature.
    
    Core features cannot be toggled (they are always enabled).
    
    Request body:
    {
        "feature_name": "pressure",
        "enabled": true
    }
    
    Response:
    {
        "status": "success",
        "message": "Feature 'pressure' is now enabled",
        "active_features": [...]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        feature_name = data.get("feature_name")
        enabled = data.get("enabled")
        
        if not feature_name:
            return jsonify({
                "status": "error",
                "message": "feature_name is required",
            }), 400
        
        if enabled is None:
            return jsonify({
                "status": "error",
                "message": "enabled is required (true or false)",
            }), 400
        
        config = get_feature_config()
        
        if enabled:
            result = config.enable_experimental_feature(feature_name)
        else:
            result = config.disable_experimental_feature(feature_name)
        
        if not result:
            return jsonify({
                "status": "error",
                "message": f"Feature '{feature_name}' is not an experimental feature (cannot toggle core features)",
            }), 400
        
        # Save configuration
        config.save()
        
        status = "enabled" if enabled else "disabled"
        return jsonify({
            "status": "success",
            "message": f"Feature '{feature_name}' is now {status}",
            "active_features": config.get_active_feature_names(),
        })
    except Exception as e:
        _Logger.error("Error toggling feature: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/features/timezone")
def set_feature_timezone():
    """
    Set the timezone for time-based features.
    
    Request body:
    {
        "timezone": "Europe/Amsterdam"
    }
    
    Response:
    {
        "status": "success",
        "message": "Timezone set to Europe/Amsterdam",
        "timezone": "Europe/Amsterdam"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        timezone = data.get("timezone")
        
        if not timezone:
            return jsonify({
                "status": "error",
                "message": "timezone is required (IANA identifier, e.g., 'Europe/Amsterdam')",
            }), 400
        
        config = get_feature_config()
        result = config.set_timezone(timezone)
        
        if not result:
            return jsonify({
                "status": "error",
                "message": f"Invalid timezone: '{timezone}'. Use IANA timezone identifiers.",
            }), 400
        
        # Save configuration
        config.save()
        
        return jsonify({
            "status": "success",
            "message": f"Timezone set to {timezone}",
            "timezone": config.timezone,
        })
    except Exception as e:
        _Logger.error("Error setting timezone: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/features/metadata")
def get_features_metadata():
    """
    Get metadata for all features.
    
    This is the single source of truth for feature documentation.
    
    Response:
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
    """
    try:
        metadata = get_feature_metadata_dict()
        
        return jsonify({
            "status": "success",
            "features": metadata,
            "core_count": get_core_feature_count(),
        })
    except Exception as e:
        _Logger.error("Error getting feature metadata: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# TWO-STEP PREDICTION ENDPOINTS (EXPERIMENTAL)
# =============================================================================


@app.get("/api/features/two_step_prediction")
def get_two_step_prediction_config():
    """
    Get the two-step prediction configuration status.
    
    Two-step prediction is an experimental feature that:
    1. First classifies whether heating will be active in a given hour
    2. Then predicts kWh consumption only for active hours
    
    Response:
    {
        "status": "success",
        "two_step_prediction_enabled": false,
        "description": "Two-step prediction: classifier + regressor for better accuracy"
    }
    """
    try:
        config = get_feature_config()
        two_step_model = _get_two_step_model()
        
        return jsonify({
            "status": "success",
            "two_step_prediction_enabled": config.is_two_step_prediction_enabled(),
            "two_step_model_available": two_step_model is not None and two_step_model.is_available,
            "activity_threshold_kwh": two_step_model.activity_threshold_kwh if two_step_model else None,
            "description": "Two-step prediction: classifier + regressor for better accuracy. "
                          "First predicts if heating is active, then predicts kWh for active hours only.",
        })
    except Exception as e:
        _Logger.error("Error getting two-step prediction config: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/features/two_step_prediction")
def toggle_two_step_prediction():
    """
    Enable or disable two-step prediction mode.
    
    Request body:
    {
        "enabled": true
    }
    
    Response:
    {
        "status": "success",
        "message": "Two-step prediction enabled",
        "two_step_prediction_enabled": true
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        enabled = data.get("enabled")
        
        if enabled is None:
            return jsonify({
                "status": "error",
                "message": "enabled is required (true or false)",
            }), 400
        
        config = get_feature_config()
        
        if enabled:
            config.enable_two_step_prediction()
        else:
            config.disable_two_step_prediction()
        
        # Save configuration
        config.save()
        
        status = "enabled" if enabled else "disabled"
        return jsonify({
            "status": "success",
            "message": f"Two-step prediction {status}",
            "two_step_prediction_enabled": config.is_two_step_prediction_enabled(),
        })
    except Exception as e:
        _Logger.error("Error toggling two-step prediction: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/train/two_step_heating_demand")
def train_two_step_heating_demand():
    """
    Train the two-step heating demand prediction model.
    
    This trains:
    1. A classifier to predict active/inactive hours
    2. A regressor to predict kWh for active hours only
    
    The activity threshold is automatically computed from the training data.
    
    Response:
    {
        "status": "success",
        "message": "Two-step model trained successfully",
        "threshold": {
            "computed_threshold_kwh": 0.05,
            "active_samples": 2500,
            "inactive_samples": 1500
        },
        "classifier_metrics": {
            "accuracy": 0.92,
            "precision": 0.88,
            "recall": 0.95,
            "f1": 0.91
        },
        "regressor_metrics": {
            "train_samples": 2000,
            "val_samples": 500,
            "train_mae_kwh": 0.15,
            "val_mae_kwh": 0.18,
            "val_mape_pct": 12.5,
            "val_r2": 0.85
        }
    }
    """
    global _two_step_model
    try:
        _Logger.info("Training two-step heating demand model...")
        
        # Build feature dataset (same as single-step model)
        df, stats = build_heating_feature_dataset(min_samples=50)
        
        if df is None:
            return jsonify({
                "status": "error",
                "message": "Insufficient data for training",
                "stats": {
                    "total_slots": stats.total_slots,
                    "valid_slots": stats.valid_slots,
                    "dropped_missing_features": stats.dropped_missing_features,
                    "dropped_missing_target": stats.dropped_missing_target,
                },
            }), 400
        
        # Train two-step model
        model, metrics = train_two_step_heating_demand_model(df)
        _two_step_model = model
        
        return jsonify({
            "status": "success",
            "message": "Two-step model trained successfully",
            "threshold": {
                "computed_threshold_kwh": round(metrics.computed_threshold_kwh, 4),
                "active_samples": metrics.active_samples,
                "inactive_samples": metrics.inactive_samples,
            },
            "classifier_metrics": {
                "accuracy": round(metrics.classifier_accuracy, 4),
                "precision": round(metrics.classifier_precision, 4),
                "recall": round(metrics.classifier_recall, 4),
                "f1": round(metrics.classifier_f1, 4),
            },
            "regressor_metrics": {
                "train_samples": metrics.regressor_train_samples,
                "val_samples": metrics.regressor_val_samples,
                "train_mae_kwh": round(metrics.regressor_train_mae, 4),
                "val_mae_kwh": round(metrics.regressor_val_mae, 4),
                "val_mape_pct": round(metrics.regressor_val_mape * 100, 2) if metrics.regressor_val_mape == metrics.regressor_val_mape else None,
                "val_r2": round(metrics.regressor_val_r2, 4),
            },
            "dataset_stats": {
                "total_slots": stats.total_slots,
                "valid_slots": stats.valid_slots,
                "features_used": stats.features_used,
            },
            "training_data": _build_training_data_response(stats),
        })
        
    except Exception as e:
        _Logger.error("Error training two-step heating demand model: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/model/two_step_status")
def get_two_step_model_status():
    """Get the status of the two-step heating demand model."""
    model = _get_two_step_model()
    
    if model is None or not model.is_available:
        return jsonify({
            "status": "not_available",
            "message": "No trained two-step model available. "
                      "Train via POST /api/train/two_step_heating_demand",
        })
    
    return jsonify({
        "status": "available",
        "activity_threshold_kwh": round(model.activity_threshold_kwh, 4),
        "features": model.feature_names,
        "training_timestamp": model.training_timestamp.isoformat() if model.training_timestamp else None,
    })


@app.post("/api/predictions/two_step_scenario")
def predict_two_step_heating_demand_scenario():
    """
    Predict heating demand using the two-step model with simplified inputs.
    
    This endpoint uses the experimental two-step approach:
    1. First classifies each hour as active or inactive
    2. For active hours, predicts kWh consumption
    3. Inactive hours are predicted as 0 kWh
    
    Request body: (same as /api/predictions/scenario)
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
            ...
        ]
    }
    
    Response:
    {
        "status": "success",
        "predictions": [
            {
                "timestamp": "2024-01-15T14:00:00",
                "is_active": true,
                "predicted_kwh": 1.2345,
                "activity_probability": 0.87
            },
            ...
        ],
        "summary": {
            "total_kwh": 24.5,
            "active_hours": 18,
            "inactive_hours": 6
        }
    }
    """
    model = _get_two_step_model()
    
    if model is None or not model.is_available:
        return jsonify({
            "status": "error",
            "message": "Two-step model not trained. "
                      "Please train via POST /api/train/two_step_heating_demand",
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        timeslots = data.get("timeslots", [])
        
        if not timeslots:
            return jsonify({
                "status": "error",
                "message": "timeslots is required and must be non-empty",
                "required_fields": SIMPLIFIED_REQUIRED_FIELDS,
                "optional_fields": SIMPLIFIED_OPTIONAL_FIELDS,
            }), 400
        
        # Validate simplified scenario input
        validation = validate_simplified_scenario(timeslots)
        
        if not validation.valid:
            return jsonify({
                "status": "error",
                "message": "Validation failed",
                "errors": validation.errors,
                "required_fields": SIMPLIFIED_REQUIRED_FIELDS,
                "optional_fields": SIMPLIFIED_OPTIONAL_FIELDS,
            }), 400
        
        # Convert simplified input to model features
        model_features, parsed_timestamps = convert_simplified_to_model_features(
            timeslots,
            model.feature_names,
            include_historical_heating=True,
        )
        
        # Make predictions using two-step approach
        predictions = predict_two_step_scenario(
            model,
            model_features,
            update_historical=True,
        )
        
        # Build response with detailed prediction info
        prediction_results = []
        active_count = 0
        inactive_count = 0
        total_kwh = 0.0
        
        for ts, pred in zip(parsed_timestamps, predictions):
            prediction_results.append({
                "timestamp": ts.isoformat(),
                "is_active": pred.is_active,
                "predicted_kwh": round(pred.predicted_kwh, PREDICTION_DECIMAL_PLACES),
                "activity_probability": round(pred.classifier_probability, 4),
            })
            
            if pred.is_active:
                active_count += 1
            else:
                inactive_count += 1
            total_kwh += pred.predicted_kwh
        
        return jsonify({
            "status": "success",
            "predictions": prediction_results,
            "summary": {
                "total_kwh": round(total_kwh, PREDICTION_DECIMAL_PLACES),
                "active_hours": active_count,
                "inactive_hours": inactive_count,
            },
            "model_info": {
                "activity_threshold_kwh": round(model.activity_threshold_kwh, 4),
                "training_timestamp": model.training_timestamp.isoformat() if model.training_timestamp else None,
            },
        })
        
    except TwoStepModelNotAvailableError:
        return jsonify({
            "status": "error",
            "message": "Two-step model not available",
        }), 503
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400
    except Exception as e:
        _Logger.error("Error predicting with two-step model: %s", e, exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


if __name__ == "__main__":
    # Initialize database schema and sensor mappings before starting workers
    init_db_schema()
    sync_sensor_mappings()
    
    # Try to load existing models
    _get_model()
    _get_two_step_model()
    
    start_sensor_logging_worker()
    app.run(host="0.0.0.0", port=8099, debug=False)
