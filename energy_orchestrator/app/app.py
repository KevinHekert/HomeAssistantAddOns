import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request
import pandas as pd
from ha.ha_api import get_entity_state
from workers import start_sensor_logging_worker
from db.resample import resample_all_categories_to_5min, resample_all_categories, get_sample_rate_minutes, VALID_SAMPLE_RATES
from db.core import init_db_schema
from db.sensor_config import sync_sensor_mappings
from db.samples import get_sensor_info
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
)
from ml.heating_demand_model import (
    HeatingDemandModel,
    ModelNotAvailableError,
    load_heating_demand_model,
    train_heating_demand_model,
    predict_scenario,
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

# Global model instance
_heating_model: HeatingDemandModel | None = None


def _get_model() -> HeatingDemandModel | None:
    """Get the global heating demand model, loading it if necessary."""
    global _heating_model
    if _heating_model is None:
        try:
            _heating_model = load_heating_demand_model()
        except Exception as e:
            _Logger.error("Failed to load heating demand model: %s", e)
    return _heating_model


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
        "sample_rate_minutes": 5  // Optional, overrides configured rate
    }
    
    Valid sample rates are: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60
    These are divisors of 60 that ensure proper hour boundary alignment.
    """
    try:
        # Check if a sample rate was provided in the request
        sample_rate = None
        if request.is_json:
            data = request.get_json()
            if data and "sample_rate_minutes" in data:
                sample_rate = int(data["sample_rate_minutes"])
                if sample_rate not in VALID_SAMPLE_RATES:
                    return jsonify({
                        "status": "error",
                        "message": f"sample_rate_minutes must be one of {VALID_SAMPLE_RATES}",
                    }), 400
        
        _Logger.info("Resample triggered via UI with sample_rate=%s", sample_rate or "default")
        
        if sample_rate is not None:
            stats = resample_all_categories(sample_rate)
        else:
            stats = resample_all_categories_to_5min()
        
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


@app.get("/api/examples/single_slot")
def get_single_slot_example():
    """Get a pre-filled example for single slot prediction."""
    model = _get_model()
    
    # Create a realistic example for a winter day
    example = {
        "outdoor_temp": 5.0,
        "wind": 3.5,
        "humidity": 75.0,
        "pressure": 1013.0,
        "indoor_temp": 19.5,
        "target_temp": 20.0,
        "hour_of_day": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_night": 0,
    }
    
    # Add model-specific features if model is available
    if model and model.is_available:
        for feature in model.feature_names:
            if feature not in example:
                # Add default values for missing features
                if "avg" in feature or "last" in feature:
                    example[feature] = 5.0  # Default average/historical value
                elif "heating_kwh" in feature:
                    example[feature] = 2.0  # Default heating energy
                else:
                    example[feature] = 0.0
    
    return jsonify({
        "status": "success",
        "example": {
            "timeslots": ["2024-01-15T14:00:00"],
            "scenario_features": [example],
            "update_historical": False,
        },
        "description": "Single 5-minute slot prediction for a typical winter afternoon",
        "model_available": model is not None and model.is_available,
        "required_features": model.feature_names if model and model.is_available else None,
    })


@app.get("/api/examples/full_day")
def get_full_day_example():
    """Get a pre-filled example for full day (24h) prediction."""
    model = _get_model()
    
    # Generate 24 hours of 5-minute slots (288 slots)
    # For simplicity in UI, we'll use 1-hour slots (24 slots)
    timeslots = []
    scenario_features = []
    
    for hour in range(24):
        # Simulate outdoor temp variation (coldest at night, warmer during day)
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
        
        is_night = 1 if (hour < 6 or hour >= 22) else 0
        
        features = {
            "outdoor_temp": round(outdoor_temp, 1),
            "wind": 3.5,
            "humidity": 75.0,
            "pressure": 1013.0,
            "indoor_temp": 19.5,
            "target_temp": target_temp,
            "hour_of_day": hour,
            "day_of_week": 2,
            "is_weekend": 0,
            "is_night": is_night,
        }
        
        # Add model-specific features if model is available
        if model and model.is_available:
            for feature in model.feature_names:
                if feature not in features:
                    if "avg" in feature or "last" in feature:
                        features[feature] = outdoor_temp
                    elif "heating_kwh" in feature:
                        features[feature] = 2.0
                    else:
                        features[feature] = 0.0
        
        timeslots.append(f"2024-01-15T{hour:02d}:00:00")
        scenario_features.append(features)
    
    return jsonify({
        "status": "success",
        "example": {
            "timeslots": timeslots,
            "scenario_features": scenario_features,
            "update_historical": True,
        },
        "description": "24-hour prediction with typical winter day temperature variation and setpoint schedule",
        "model_available": model is not None and model.is_available,
        "required_features": model.feature_names if model and model.is_available else None,
    })


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
        
        # Make predictions
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
            },
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
        # Note: timestamps remain historical for comparison purposes
        scenario_timeslots = []
        for hour in data:
            slot = {
                "timestamp": hour["timestamp"],
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


if __name__ == "__main__":
    # Initialize database schema and sensor mappings before starting workers
    init_db_schema()
    sync_sensor_mappings()
    
    # Try to load existing model
    _get_model()
    
    start_sensor_logging_worker()
    app.run(host="0.0.0.0", port=8099, debug=False)
