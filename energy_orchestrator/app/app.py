import logging
from flask import Flask, render_template, jsonify, request
from ha.ha_api import get_entity_state
from workers import start_sensor_logging_worker
from db.resample import resample_all_categories_to_5min
from db.core import init_db_schema
from db.sensor_config import sync_sensor_mappings
from db.samples import get_sensor_info
from ml.heating_features import build_heating_feature_dataset
from ml.heating_demand_model import (
    HeatingDemandModel,
    ModelNotAvailableError,
    load_heating_demand_model,
    train_heating_demand_model,
    predict_scenario,
)


app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
_Logger = logging.getLogger(__name__)

# Voor nu één sensor; later lijst vanuit config/integratie
WIND_ENTITY_ID = "sensor.knmi_windsnelheid"

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
    """Trigger resampling of all categories to 5-minute slots."""
    try:
        _Logger.info("Resample triggered via UI")
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
            },
        })
    except Exception as e:
        _Logger.error("Error during resampling: %s", e)
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


if __name__ == "__main__":
    # Initialize database schema and sensor mappings before starting workers
    init_db_schema()
    sync_sensor_mappings()
    
    # Try to load existing model
    _get_model()
    
    start_sensor_logging_worker()
    app.run(host="0.0.0.0", port=8099, debug=False)
