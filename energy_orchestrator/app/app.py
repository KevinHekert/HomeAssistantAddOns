import logging
import threading
from datetime import datetime, timedelta
from typing import Optional
from flask import Flask, render_template, jsonify, request
from sqlalchemy.orm import Session
import pandas as pd

# Thread lock for optimizer state
_optimizer_lock = threading.Lock()
from ha.ha_api import get_entity_state
from ha.weather_api import (
    get_weather_config,
    set_weather_config,
    validate_weather_api,
    fetch_weather_forecast,
    convert_forecast_to_scenario_timeslots,
)
from workers import start_sensor_logging_worker
from db.resample import resample_all_categories, get_sample_rate_minutes, set_sample_rate_minutes, VALID_SAMPLE_RATES, flush_resampled_samples
from db.core import init_db_schema, engine
from db import ResampledSample, FeatureStatistic
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
from db.sensor_category_config import (
    get_sensor_category_config,
    get_all_sensor_definitions,
    get_sensor_definition,
    CORE_SENSORS,
    EXPERIMENTAL_SENSORS,
)
from db.virtual_sensors import (
    get_virtual_sensors_config,
    VirtualSensorDefinition,
    VirtualSensorOperation,
)
from db.feature_stats import (
    get_feature_stats_config,
    StatType,
)
from db.calculate_feature_stats import (
    calculate_feature_statistics,
    flush_feature_statistics,
)
from db.prediction_storage import (
    store_prediction,
    get_stored_predictions,
    get_prediction_by_id,
    delete_prediction,
    compare_prediction_with_actual,
    get_prediction_list_summary,
)
from db.optimizer_storage import (
    save_optimizer_run,
    get_latest_optimizer_run,
    get_optimizer_run_by_id,
    get_optimizer_result_by_id,
    list_optimizer_runs,
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
    FeatureMetadata,
    categorize_features,
    get_feature_details,
    verify_model_features,
    CORE_FEATURES,
    EXPERIMENTAL_FEATURES,
)
from ml.optimizer import (
    run_optimization,
    apply_best_configuration,
    OptimizerProgress,
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
    
    After resampling completes, feature statistics (time-span averages) are
    automatically calculated from the resampled data.
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
        
        # Step 1: Always use resample_all_categories - it uses configured rate when sample_rate is None
        stats = resample_all_categories(sample_rate, flush=flush)
        
        # Step 2: Automatically calculate feature statistics after successful resampling
        feature_stats_result = None
        try:
            _Logger.info("Calculating feature statistics after resampling...")
            # If we flushed resampled data, also flush feature statistics
            if flush:
                flush_feature_statistics()
                _Logger.info("Flushed feature statistics due to flush=True")
            
            feature_stats_result = calculate_feature_statistics()
            _Logger.info(
                "Feature statistics calculated: %d stats saved for %d sensors",
                feature_stats_result.stats_saved,
                feature_stats_result.sensors_processed,
            )
        except Exception as e:
            # Log error but don't fail the entire request
            # Resampling was successful, feature stats calculation is secondary
            _Logger.error("Error calculating feature statistics: %s", e, exc_info=True)
        
        response = {
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
        }
        
        # Add feature statistics info to response if calculation succeeded
        if feature_stats_result:
            response["feature_stats"] = {
                "stats_calculated": feature_stats_result.stats_calculated,
                "stats_saved": feature_stats_result.stats_saved,
                "sensors_processed": feature_stats_result.sensors_processed,
                "stat_types": feature_stats_result.stat_types_processed,
            }
        
        return jsonify(response)
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


@app.get("/api/resampled_data")
def get_resampled_data():
    """
    Query resampled data with is_derived field information.
    
    Query parameters:
    - category (optional): Filter by sensor category/name
    - start_time (optional): ISO format datetime string
    - end_time (optional): ISO format datetime string
    - limit (optional): Maximum number of records (default: 100, max: 1000)
    - is_derived (optional): Filter by is_derived flag (true/false)
    
    Response:
    {
        "status": "success",
        "data": [
            {
                "slot_start": "2024-12-02T10:00:00",
                "category": "outdoor_temp",
                "value": 5.5,
                "unit": "°C",
                "is_derived": false
            },
            ...
        ],
        "count": 100,
        "has_more": true
    }
    """
    try:
        # Parse query parameters
        category = request.args.get('category')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        limit = min(int(request.args.get('limit', 100)), 1000)
        is_derived_str = request.args.get('is_derived')
        
        # Build query
        with Session(engine) as session:
            query = session.query(ResampledSample)
            
            if category:
                query = query.filter(ResampledSample.category == category)
            
            if start_time:
                start_dt = datetime.fromisoformat(start_time)
                query = query.filter(ResampledSample.slot_start >= start_dt)
            
            if end_time:
                end_dt = datetime.fromisoformat(end_time)
                query = query.filter(ResampledSample.slot_start <= end_dt)
            
            if is_derived_str is not None:
                is_derived = is_derived_str.lower() == 'true'
                query = query.filter(ResampledSample.is_derived == is_derived)
            
            # Order by slot_start descending (most recent first)
            query = query.order_by(ResampledSample.slot_start.desc())
            
            # Get one more than limit to check if there are more
            results = query.limit(limit + 1).all()
            
            has_more = len(results) > limit
            if has_more:
                results = results[:limit]
            
            # Convert to dict
            data = [
                {
                    "slot_start": r.slot_start.isoformat(),
                    "category": r.category,
                    "value": r.value,
                    "unit": r.unit,
                    "is_derived": r.is_derived,
                }
                for r in results
            ]
            
            return jsonify({
                "status": "success",
                "data": data,
                "count": len(data),
                "has_more": has_more,
            })
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid parameter: {str(e)}",
        }), 400
    except Exception as e:
        _Logger.error("Error querying resampled data: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/feature_statistics")
def get_feature_statistics_data():
    """
    Query feature statistics (time-span averages) from the feature_statistics table.
    
    Query parameters:
    - sensor_name (optional): Filter by sensor name
    - stat_type (optional): Filter by statistic type (avg_1h, avg_6h, avg_24h, avg_7d)
    - start_time (optional): ISO format datetime string
    - end_time (optional): ISO format datetime string
    - limit (optional): Maximum number of records (default: 100, max: 1000)
    
    Response:
    {
        "status": "success",
        "data": [
            {
                "slot_start": "2024-12-02T10:00:00",
                "sensor_name": "outdoor_temp",
                "stat_type": "avg_1h",
                "value": 5.5,
                "unit": "°C",
                "source_sample_count": 12
            },
            ...
        ],
        "count": 100,
        "has_more": true
    }
    """
    try:
        # Parse query parameters
        sensor_name = request.args.get('sensor_name')
        stat_type = request.args.get('stat_type')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        limit = min(int(request.args.get('limit', 100)), 1000)
        
        # Build query
        with Session(engine) as session:
            query = session.query(FeatureStatistic)
            
            if sensor_name:
                query = query.filter(FeatureStatistic.sensor_name == sensor_name)
            
            if stat_type:
                query = query.filter(FeatureStatistic.stat_type == stat_type)
            
            if start_time:
                start_dt = datetime.fromisoformat(start_time)
                query = query.filter(FeatureStatistic.slot_start >= start_dt)
            
            if end_time:
                end_dt = datetime.fromisoformat(end_time)
                query = query.filter(FeatureStatistic.slot_start <= end_dt)
            
            # Order by slot_start descending (most recent first)
            query = query.order_by(FeatureStatistic.slot_start.desc())
            
            # Get one more than limit to check if there are more
            results = query.limit(limit + 1).all()
            
            has_more = len(results) > limit
            if has_more:
                results = results[:limit]
            
            # Convert to dict
            data = [
                {
                    "slot_start": r.slot_start.isoformat(),
                    "sensor_name": r.sensor_name,
                    "stat_type": r.stat_type,
                    "value": r.value,
                    "unit": r.unit,
                    "source_sample_count": r.source_sample_count,
                }
                for r in results
            ]
            
            return jsonify({
                "status": "success",
                "data": data,
                "count": len(data),
                "has_more": has_more,
            })
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid parameter: {str(e)}",
        }), 400
    except Exception as e:
        _Logger.error("Error querying feature statistics: %s", e)
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


def _build_training_data_response(stats: FeatureDatasetStats) -> dict:
    """
    Build the training_data response with all sensor categories.
    
    For hp_kwh_total, shows the delta (energy consumed) instead of raw cumulative values.
    Unit information is extracted from the actual data in the samples table.
    """
    training_data = {}
    
    for category, range_data in stats.sensor_ranges.items():
        if category == "hp_kwh_total":
            # Show delta instead of first/last cumulative values
            training_data[category] = {
                "delta": stats.hp_kwh_delta,
                "first": range_data.first,
                "last": range_data.last,
                "unit": range_data.unit or "",
            }
        else:
            training_data[category] = {
                "first": range_data.first,
                "last": range_data.last,
                "unit": range_data.unit or "",
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
        
        # Verify features are actually used
        feature_verification = verify_model_features(
            model_feature_names=metrics.features,
            dataset_feature_names=stats.features_used,
        )
        
        # Categorize features as raw vs calculated
        feature_categories = categorize_features(metrics.features)
        
        # Get detailed feature info
        feature_details = get_feature_details(metrics.features)
        
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
            "feature_verification": {
                "verified": feature_verification["verified"],
                "feature_count": feature_verification["feature_count"],
                "verified_features": feature_verification["verified_features"],
                "missing_in_dataset": feature_verification["missing_in_dataset"],
                "message": "All features verified as used in training" if feature_verification["verified"] else "Some features are missing from dataset",
            },
            "feature_categories": feature_categories,
            "feature_details": feature_details,
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
        if use_two_step:
            return jsonify({
                "status": "error",
                "message": "Two-step model not trained. Please train the model first via POST /api/train/two_step_heating_demand",
            }), 503
        else:
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
def toggle_feature():
    """
    Enable or disable any feature (core or experimental).
    
    Both core and experimental features can be toggled.
    Core features are labeled as 'CORE' in UI but can still be disabled.
    
    Request body:
    {
        "feature_name": "pressure",
        "enabled": true
    }
    
    Response:
    {
        "status": "success",
        "message": "Feature 'pressure' is now enabled",
        "active_features": [...],
        "is_core": false
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
        
        # Try to toggle the feature (works for both core and experimental)
        if enabled:
            result = config.enable_feature(feature_name)
        else:
            result = config.disable_feature(feature_name)
        
        if not result:
            return jsonify({
                "status": "error",
                "message": f"Feature '{feature_name}' not found",
            }), 404
        
        # Determine if it's a core feature
        is_core = any(f.name == feature_name for f in CORE_FEATURES)
        
        # Save configuration
        config.save()
        
        # Note: We do NOT sync feature stats configuration here.
        # Sensor stats configuration should remain independent from ML feature configuration.
        # Users configure sensor stats separately to collect data, and ML features determine
        # which features are used for training, not which stats are collected.
        
        status = "enabled" if enabled else "disabled"
        return jsonify({
            "status": "success",
            "message": f"Feature '{feature_name}' is now {status}",
            "active_features": config.get_active_feature_names(),
            "is_core": is_core,
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


@app.get("/api/features/sensors_with_stats")
def get_sensors_with_statistics():
    """
    Get comprehensive sensor information including their time-based statistics.
    
    This endpoint provides all sensors (raw + virtual) with their enabled statistics
    for display in the Feature Configuration section.
    
    Response:
    {
        "status": "success",
        "sensors": [
            {
                "name": "outdoor_temp",
                "display_name": "Outdoor Temperature",
                "type": "raw",
                "enabled": true,
                "enabled_stats": ["avg_1h", "avg_6h", "avg_24h"],
                "stat_features": [
                    {"name": "outdoor_temp_avg_1h", "type": "avg_1h"},
                    {"name": "outdoor_temp_avg_6h", "type": "avg_6h"},
                    ...
                ]
            },
            {
                "name": "temp_delta",
                "display_name": "Temperature Delta",
                "type": "virtual",
                "enabled": true,
                "description": "Target - Indoor",
                "enabled_stats": ["avg_1h"],
                ...
            }
        ],
        "total_count": 15,
        "raw_count": 10,
        "virtual_count": 5
    }
    """
    try:
        sensor_category_config = get_sensor_category_config()
        virtual_sensors_config = get_virtual_sensors_config()
        feature_stats_config = get_feature_stats_config()
        
        sensors_list = []
        raw_count = 0
        virtual_count = 0
        
        # Add raw sensors
        for sensor_config in sensor_category_config.get_enabled_sensors():
            # Get the sensor definition to access display_name
            sensor_def = get_sensor_definition(sensor_config.category_name)
            display_name = sensor_def.display_name if sensor_def else sensor_config.category_name
            
            enabled_stats = feature_stats_config.get_enabled_stats_for_sensor(sensor_config.category_name)
            stat_features = [
                {
                    "name": feature_stats_config.get_sensor_config(sensor_config.category_name).get_stat_category_name(stat),
                    "type": stat.value
                }
                for stat in enabled_stats
            ]
            
            sensors_list.append({
                "name": sensor_config.category_name,
                "display_name": display_name,
                "type": "raw",
                "enabled": sensor_config.enabled,
                "enabled_stats": [s.value for s in enabled_stats],
                "stat_features": stat_features,
            })
            raw_count += 1
        
        # Add virtual sensors
        for virtual_sensor in virtual_sensors_config.get_enabled_sensors():
            enabled_stats = feature_stats_config.get_enabled_stats_for_sensor(virtual_sensor.name)
            stat_features = [
                {
                    "name": feature_stats_config.get_sensor_config(virtual_sensor.name).get_stat_category_name(stat),
                    "type": stat.value
                }
                for stat in enabled_stats
            ]
            
            sensors_list.append({
                "name": virtual_sensor.name,
                "display_name": virtual_sensor.display_name,
                "type": "virtual",
                "enabled": virtual_sensor.enabled,
                "description": virtual_sensor.description,
                "operation": virtual_sensor.operation.value,
                "enabled_stats": [s.value for s in enabled_stats],
                "stat_features": stat_features,
            })
            virtual_count += 1
        
        return jsonify({
            "status": "success",
            "sensors": sensors_list,
            "total_count": len(sensors_list),
            "raw_count": raw_count,
            "virtual_count": virtual_count,
        })
    except Exception as e:
        _Logger.error("Error getting sensors with statistics: %s", e)
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


@app.get("/api/features/special_cards")
def get_special_feature_cards():
    """
    Get special feature cards for Time/Date and Usage calculated features.
    
    These are system-generated cards that group calculated features which are
    not directly tied to physical sensors.
    
    Response:
    {
        "status": "success",
        "cards": [
            {
                "id": "time_date",
                "name": "Time & Date Features",
                "description": "Calendar and time-based features for temporal patterns",
                "type": "calculated",
                "features": [
                    {
                        "name": "hour_of_day",
                        "display_name": "Hour of Day",
                        "description": "Local hour (0-23) in configured timezone",
                        "unit": "hour",
                        "is_core": true,
                        "enabled": true
                    },
                    ...
                ]
            },
            {
                "id": "usage_heating",
                "name": "Heating Usage Features",
                "description": "Historical heating consumption and demand metrics",
                "type": "calculated",
                "features": [...]
            }
        ]
    }
    """
    try:
        config = get_feature_config()
        
        # Time & Date Features
        time_date_features = ["hour_of_day", "day_of_week", "is_weekend", "is_night"]
        
        # Usage/Heating Features
        usage_features = [
            "heating_kwh_last_1h",
            "heating_kwh_last_6h", 
            "heating_kwh_last_24h",
            "heating_kwh_last_7d",
            "heating_degree_hours_24h",
            "heating_degree_hours_7d",
        ]
        
        # Get all features with their metadata
        all_features = config.get_all_features()
        features_dict = {f.name: f for f in all_features}
        
        # Build Time/Date card
        time_date_card = {
            "id": "time_date",
            "name": "Time & Date Features",
            "description": "Calendar and time-based features for temporal patterns",
            "type": "calculated",
            "features": []
        }
        
        for feature_name in time_date_features:
            if feature_name in features_dict:
                f = features_dict[feature_name]
                time_date_card["features"].append({
                    "name": f.name,
                    "display_name": f.name.replace("_", " ").title(),
                    "description": f.description,
                    "unit": f.unit,
                    "is_core": f.is_core,
                    "enabled": f.enabled,
                })
        
        # Build Usage card
        usage_card = {
            "id": "usage_heating",
            "name": "Heating Usage Features",
            "description": "Historical heating consumption and demand metrics",
            "type": "calculated",
            "features": []
        }
        
        for feature_name in usage_features:
            if feature_name in features_dict:
                f = features_dict[feature_name]
                usage_card["features"].append({
                    "name": f.name,
                    "display_name": f.name.replace("_", " ").title(),
                    "description": f.description,
                    "unit": f.unit,
                    "is_core": f.is_core,
                    "enabled": f.enabled,
                })
        
        return jsonify({
            "status": "success",
            "cards": [time_date_card, usage_card]
        })
    except Exception as e:
        _Logger.error("Error getting special feature cards: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/features/sensor_cards")
def get_sensor_feature_cards():
    """
    Get feature cards for physical sensors with their aggregation features.
    
    Returns sensor cards showing which time-based aggregations (avg_1h, avg_6h, etc.)
    are available as features for each sensor.
    
    Response:
    {
        "status": "success",
        "sensor_cards": [
            {
                "sensor_name": "outdoor_temp",
                "display_name": "Outdoor Temperature",
                "unit": "°C",
                "type": "weather",
                "features": [
                    {
                        "name": "outdoor_temp",
                        "display_name": "Current Value",
                        "is_core": true,
                        "enabled": true
                    },
                    {
                        "name": "outdoor_temp_avg_1h",
                        "display_name": "1-Hour Average",
                        "is_core": true,
                        "enabled": true
                    },
                    ...
                ]
            },
            ...
        ]
    }
    """
    try:
        config = get_feature_config()
        all_features = config.get_all_features()
        
        # Get sensor information from sensor_category_config
        sensor_category_conf = get_sensor_category_config()
        all_sensor_defs = get_all_sensor_definitions()
        virtual_sensors_conf = get_virtual_sensors_config()
        
        # Get feature stats configuration to determine which time-based stats are enabled
        feature_stats_conf = get_feature_stats_config()
        
        # Build dynamic sensor metadata from configured sensors
        raw_sensors: dict[str, dict] = {}
        
        # Add all configured raw sensors (both core and experimental, enabled or disabled)
        # We show all sensors so users can see what's available and configure them
        for sensor_def in all_sensor_defs:
            sensor_config = sensor_category_conf.get_sensor_config(sensor_def.category_name)
            # sensor_config should always exist since __post_init__ creates defaults,
            # but we check to be safe in case of configuration corruption
            if sensor_config:
                raw_sensors[sensor_def.category_name] = {
                    "display_name": sensor_def.display_name,
                    "unit": sensor_config.unit if sensor_config.unit else sensor_def.unit,
                    "type": sensor_def.sensor_type.value,
                }
        
        # Add all virtual sensors (enabled or disabled)
        for virtual_sensor in virtual_sensors_conf.sensors.values():
            raw_sensors[virtual_sensor.name] = {
                "display_name": virtual_sensor.display_name,
                "unit": virtual_sensor.unit,
                "type": "virtual",  # Virtual sensors get their own type
            }
        
        # Build a lookup map of feature names to feature objects
        feature_map: dict[str, FeatureMetadata] = {f.name: f for f in all_features}
        
        # Group features by base sensor name
        sensor_features: dict[str, list] = {}
        
        # Initialize sensor groups
        for sensor_name in raw_sensors.keys():
            sensor_features[sensor_name] = []
        
        # For each sensor, add the base sensor feature and all configured time-based statistics
        for sensor_name in raw_sensors.keys():
            sensor_info = raw_sensors[sensor_name]
            
            # Add the base sensor feature (current value) if it exists
            if sensor_name in feature_map:
                f = feature_map[sensor_name]
                sensor_features[sensor_name].append({
                    "name": f.name,
                    "display_name": _get_feature_display_name(f.name, sensor_name),
                    "description": f.description,
                    "unit": f.unit,
                    "time_window": f.time_window.value,
                    "is_core": f.is_core,
                    "enabled": f.enabled,
                })
            else:
                # Base sensor doesn't exist in feature config yet, create it as an experimental feature
                # This happens for virtual sensors or sensors that haven't been added to feature config
                sensor_features[sensor_name].append({
                    "name": sensor_name,
                    "display_name": _get_feature_display_name(sensor_name, sensor_name),
                    "description": f"Current value for {sensor_name}",
                    "unit": sensor_info["unit"],
                    "time_window": "none",
                    "is_core": False,
                    "enabled": False,  # Not enabled in feature config yet
                })
            
            # Get enabled time-based statistics from feature stats configuration
            enabled_stats = feature_stats_conf.get_enabled_stats_for_sensor(sensor_name)
            
            # Add features for each enabled statistic
            for stat_type in enabled_stats:
                stat_feature_name = f"{sensor_name}_{stat_type.value}"
                
                # Check if this feature exists in feature configuration
                if stat_feature_name in feature_map:
                    # Feature already exists in config, use its properties
                    f = feature_map[stat_feature_name]
                    sensor_features[sensor_name].append({
                        "name": f.name,
                        "display_name": _get_feature_display_name(f.name, sensor_name),
                        "description": f.description,
                        "unit": f.unit,
                        "time_window": f.time_window.value,
                        "is_core": f.is_core,
                        "enabled": f.enabled,
                    })
                else:
                    # Feature doesn't exist in feature config yet, create it as an experimental feature
                    # This happens when a statistic is enabled in feature stats but not yet in feature config
                    sensor_features[sensor_name].append({
                        "name": stat_feature_name,
                        "display_name": _get_feature_display_name(stat_feature_name, sensor_name),
                        "description": f"Time-based statistic: {stat_type.value} for {sensor_name}",
                        "unit": sensor_info["unit"],
                        "time_window": _stat_type_to_time_window(stat_type),
                        "is_core": False,
                        "enabled": False,  # Not enabled in feature config yet
                    })
        
        # Build sensor cards
        sensor_cards = []
        for sensor_name, features in sensor_features.items():
            if features:  # Only include sensors that have features
                info = raw_sensors[sensor_name]
                sensor_cards.append({
                    "sensor_name": sensor_name,
                    "display_name": info["display_name"],
                    "unit": info["unit"],
                    "type": info["type"],
                    "features": features,
                })
        
        return jsonify({
            "status": "success",
            "sensor_cards": sensor_cards
        })
    except Exception as e:
        _Logger.error("Error getting sensor feature cards: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


def _get_feature_display_name(feature_name: str, sensor_name: str) -> str:
    """Helper to generate display names for features."""
    if feature_name == sensor_name:
        return "Current Value"
    elif "_avg_" in feature_name:
        window = feature_name.split("_avg_")[-1]
        return f"{window.upper()} Average"
    else:
        return feature_name.replace("_", " ").title()


def _stat_type_to_time_window(stat_type: StatType) -> str:
    """
    Convert StatType to TimeWindow string value.
    
    Args:
        stat_type: The StatType enum value
        
    Returns:
        TimeWindow string value (e.g., "1h", "6h", "24h", "7d")
    """
    mapping = {
        StatType.AVG_1H: "1h",
        StatType.AVG_6H: "6h",
        StatType.AVG_24H: "24h",
        StatType.AVG_7D: "7d",
    }
    return mapping.get(stat_type, "none")


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
        
        # Verify features are actually used
        feature_verification = verify_model_features(
            model_feature_names=metrics.features,
            dataset_feature_names=stats.features_used,
        )
        
        # Categorize features as raw vs calculated
        feature_categories = categorize_features(metrics.features)
        
        # Get detailed feature info
        feature_details = get_feature_details(metrics.features)
        
        return jsonify({
            "status": "success",
            "message": "Two-step model trained successfully",
            "threshold": {
                "computed_threshold_kwh": round(metrics.computed_threshold_kwh, 4),
                "active_samples": metrics.active_samples,
                "inactive_samples": metrics.inactive_samples,
            },
            # Top-level metrics for UI compatibility
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
            "step1_classifier": {
                "description": metrics.classifier_description,
                "purpose": "Predicts whether heating will be active (on) or inactive (off) for each hour",
                "features_used": metrics.features,
                "feature_count": len(metrics.features),
                "training_samples": metrics.active_samples + metrics.inactive_samples,
                "metrics": {
                    "accuracy": round(metrics.classifier_accuracy, 4),
                    "precision": round(metrics.classifier_precision, 4),
                    "recall": round(metrics.classifier_recall, 4),
                    "f1": round(metrics.classifier_f1, 4),
                },
            },
            "step2_regressor": {
                "description": metrics.regressor_description,
                "purpose": "Predicts kWh consumption for active hours only (inactive hours automatically return 0 kWh)",
                "features_used": metrics.features,
                "feature_count": len(metrics.features),
                "training_samples": metrics.regressor_train_samples,
                "note": "Only trained on active samples (samples above activity threshold)",
                "metrics": {
                    "train_samples": metrics.regressor_train_samples,
                    "val_samples": metrics.regressor_val_samples,
                    "train_mae_kwh": round(metrics.regressor_train_mae, 4),
                    "val_mae_kwh": round(metrics.regressor_val_mae, 4),
                    "val_mape_pct": round(metrics.regressor_val_mape * 100, 2) if metrics.regressor_val_mape == metrics.regressor_val_mape else None,
                    "val_r2": round(metrics.regressor_val_r2, 4),
                },
            },
            "feature_verification": {
                "verified": feature_verification["verified"],
                "feature_count": feature_verification["feature_count"],
                "verified_features": feature_verification["verified_features"],
                "missing_in_dataset": feature_verification["missing_in_dataset"],
                "message": "All features verified as used in both steps" if feature_verification["verified"] else "Some features are missing from dataset",
            },
            "feature_categories": feature_categories,
            "feature_details": feature_details,
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


# =============================================================================
# WEATHER API ENDPOINTS
# =============================================================================


@app.get("/api/weather/config")
def get_weather_config_api():
    """Get the current weather API configuration.
    
    Response:
    {
        "status": "success",
        "config": {
            "api_key": "***" (masked for security),
            "location": "Amsterdam",
            "has_api_key": true
        }
    }
    """
    try:
        config = get_weather_config()
        return jsonify({
            "status": "success",
            "config": {
                "api_key": "***" if config.api_key else "",
                "location": config.location,
                "has_api_key": bool(config.api_key),
            },
        })
    except Exception as e:
        _Logger.error("Error getting weather config: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/weather/config")
def update_weather_config():
    """Update weather API configuration.
    
    Also validates the API key and location before saving.
    
    Request body:
    {
        "api_key": "your_api_key",
        "location": "Amsterdam"
    }
    
    Response:
    {
        "status": "success",
        "message": "Weather configuration saved",
        "location_name": "Amsterdam"
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
                "message": "API key and location are required",
            }), 400
        
        api_key = data.get("api_key", "").strip()
        location = data.get("location", "").strip()
        
        if not api_key:
            return jsonify({
                "status": "error",
                "message": "API key is required",
            }), 400
        
        if not location:
            return jsonify({
                "status": "error",
                "message": "Location is required",
            }), 400
        
        # Validate the API credentials before saving
        is_valid, error_msg, location_name = validate_weather_api(api_key, location)
        
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": f"Validation failed: {error_msg}",
            }), 400
        
        # Save configuration
        success, save_error = set_weather_config(api_key=api_key, location=location)
        
        if not success:
            return jsonify({
                "status": "error",
                "message": save_error or "Failed to save configuration",
            }), 500
        
        _Logger.info("Weather config saved for location: %s (%s)", location, location_name)
        
        return jsonify({
            "status": "success",
            "message": "Weather configuration saved",
            "location_name": location_name,
        })
        
    except Exception as e:
        _Logger.error("Error updating weather config: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/weather/validate")
def validate_weather_api_endpoint():
    """Validate weather API credentials without saving.
    
    Request body:
    {
        "api_key": "your_api_key",
        "location": "Amsterdam"
    }
    
    Response:
    {
        "status": "success",
        "valid": true,
        "location_name": "Amsterdam"
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        data = request.get_json()
        api_key = data.get("api_key", "").strip()
        location = data.get("location", "").strip()
        
        is_valid, error_msg, location_name = validate_weather_api(api_key, location)
        
        if is_valid:
            return jsonify({
                "status": "success",
                "valid": True,
                "location_name": location_name,
            })
        else:
            return jsonify({
                "status": "success",
                "valid": False,
                "message": error_msg,
            })
        
    except Exception as e:
        _Logger.error("Error validating weather API: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/weather/forecast")
def get_weather_forecast():
    """Fetch weather forecast for the next 24 hours.
    
    Uses stored API configuration. Returns forecast data ready for
    use in scenario predictions.
    
    Optional query parameter:
        target_temperature: Target/setpoint temperature (default: 20.0)
    
    Response:
    {
        "status": "success",
        "location_name": "Amsterdam",
        "current_temp": 12.5,
        "hourly_forecasts": [...],
        "scenario_timeslots": [...],
        "forecast_count": 24
    }
    """
    try:
        target_temp = request.args.get("target_temperature", type=float, default=20.0)
        
        result = fetch_weather_forecast()
        
        if not result.success:
            return jsonify({
                "status": "error",
                "message": result.error_message or "Failed to fetch weather forecast",
            }), 400
        
        # Convert forecasts to JSON-serializable format
        hourly_data = []
        for forecast in result.hourly_forecasts:
            hourly_data.append({
                "timestamp": forecast.timestamp.isoformat(),
                "temperature": round(forecast.temperature, 1),
                "wind_speed": round(forecast.wind_speed, 1),
                "humidity": round(forecast.humidity, 1),
                "pressure": round(forecast.pressure, 1),
                "precipitation": round(forecast.precipitation, 1),
                "description": forecast.description,
            })
        
        # Convert to scenario format
        scenario_timeslots = convert_forecast_to_scenario_timeslots(
            result.hourly_forecasts,
            target_temperature=target_temp,
        )
        
        return jsonify({
            "status": "success",
            "location_name": result.location_name,
            "current_temp": result.current_temp,
            "hourly_forecasts": hourly_data,
            "scenario_timeslots": scenario_timeslots,
            "forecast_count": len(result.hourly_forecasts),
        })
        
    except Exception as e:
        _Logger.error("Error fetching weather forecast: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# PREDICTION STORAGE ENDPOINTS
# =============================================================================


@app.post("/api/predictions/store")
def store_prediction_endpoint():
    """Store a prediction for later comparison with actual data.
    
    Request body:
    {
        "timeslots": [...],
        "predictions": [...],
        "total_kwh": 24.5,
        "source": "weerlive",
        "location": "Amsterdam",
        "model_type": "single_step"
    }
    
    Response:
    {
        "status": "success",
        "prediction_id": "20241202_143000_123456",
        "message": "Prediction stored successfully"
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
                "message": "Request body required",
            }), 400
        
        timeslots = data.get("timeslots", [])
        predictions = data.get("predictions", [])
        total_kwh = data.get("total_kwh", 0.0)
        source = data.get("source", "manual")
        location = data.get("location", "")
        model_type = data.get("model_type", "single_step")
        
        if not predictions:
            return jsonify({
                "status": "error",
                "message": "predictions array is required",
            }), 400
        
        success, error_msg, prediction_id = store_prediction(
            timeslots=timeslots,
            predictions=predictions,
            total_kwh=total_kwh,
            source=source,
            location=location,
            model_type=model_type,
        )
        
        if success:
            return jsonify({
                "status": "success",
                "prediction_id": prediction_id,
                "message": "Prediction stored successfully",
            })
        else:
            return jsonify({
                "status": "error",
                "message": error_msg or "Failed to store prediction",
            }), 500
        
    except Exception as e:
        _Logger.error("Error storing prediction: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/predictions/stored")
def get_stored_predictions_endpoint():
    """Get list of stored predictions.
    
    Response:
    {
        "status": "success",
        "predictions": [
            {
                "id": "20241202_143000_123456",
                "created_at": "2024-12-02T14:30:00",
                "source": "weerlive",
                "location": "Amsterdam",
                "total_kwh": 24.5,
                "slots_count": 24,
                "model_type": "single_step"
            },
            ...
        ],
        "count": 5
    }
    """
    try:
        summaries = get_prediction_list_summary()
        return jsonify({
            "status": "success",
            "predictions": summaries,
            "count": len(summaries),
        })
    except Exception as e:
        _Logger.error("Error getting stored predictions: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/predictions/stored/<prediction_id>")
def get_stored_prediction_endpoint(prediction_id: str):
    """Get a specific stored prediction.
    
    Response:
    {
        "status": "success",
        "prediction": {
            "id": "20241202_143000_123456",
            "created_at": "2024-12-02T14:30:00",
            "source": "weerlive",
            "location": "Amsterdam",
            "timeslots": [...],
            "predictions": [...],
            "total_kwh": 24.5,
            "model_type": "single_step"
        }
    }
    """
    try:
        prediction = get_prediction_by_id(prediction_id)
        if not prediction:
            return jsonify({
                "status": "error",
                "message": f"Prediction not found: {prediction_id}",
            }), 404
        
        return jsonify({
            "status": "success",
            "prediction": {
                "id": prediction.id,
                "created_at": prediction.created_at,
                "source": prediction.source,
                "location": prediction.location,
                "timeslots": prediction.timeslots,
                "predictions": prediction.predictions,
                "total_kwh": prediction.total_kwh,
                "model_type": prediction.model_type,
            },
        })
    except Exception as e:
        _Logger.error("Error getting stored prediction: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.delete("/api/predictions/stored/<prediction_id>")
def delete_stored_prediction_endpoint(prediction_id: str):
    """Delete a stored prediction.
    
    Response:
    {
        "status": "success",
        "message": "Prediction deleted"
    }
    """
    try:
        if delete_prediction(prediction_id):
            return jsonify({
                "status": "success",
                "message": "Prediction deleted",
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Prediction not found: {prediction_id}",
            }), 404
    except Exception as e:
        _Logger.error("Error deleting prediction: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/predictions/stored/<prediction_id>/compare")
def compare_stored_prediction_endpoint(prediction_id: str):
    """Compare a stored prediction with actual sensor data.
    
    This endpoint retrieves actual heating kWh data for the prediction
    time range and compares it with the stored predictions.
    
    Response:
    {
        "status": "success",
        "comparison": [
            {
                "timestamp": "2024-12-02T14:00:00",
                "predicted_kwh": 1.25,
                "actual_kwh": 1.18,
                "delta_kwh": 0.07,
                "delta_pct": 5.9,
                "has_actual": true
            },
            ...
        ],
        "summary": {
            "total_predicted_kwh": 24.5,
            "total_actual_kwh": 23.8,
            "slots_compared": 20,
            "slots_missing_actual": 4,
            "mae_kwh": 0.15,
            "mape_pct": 6.2
        }
    }
    """
    try:
        prediction = get_prediction_by_id(prediction_id)
        if not prediction:
            return jsonify({
                "status": "error",
                "message": f"Prediction not found: {prediction_id}",
            }), 404
        
        # Get actual data for comparison
        # We need to get actual heating kWh for each hour in the prediction
        comparison_results = []
        total_predicted = 0.0
        total_actual = 0.0
        compared_count = 0
        abs_errors = []
        
        for pred_data in prediction.predictions:
            timestamp = pred_data.get("timestamp", "")
            predicted_kwh = pred_data.get("predicted_kwh", 0.0)
            total_predicted += predicted_kwh
            
            # Try to get actual data for this timestamp
            actual_kwh = None
            delta_kwh = None
            delta_pct = None
            has_actual = False
            
            try:
                if timestamp:
                    # Parse timestamp and get hourly data
                    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    start = ts
                    end = ts + timedelta(hours=1)
                    
                    # Use get_actual_vs_predicted_data to get actual values
                    actual_df, error = get_actual_vs_predicted_data(
                        start, end, slot_duration_minutes=60
                    )
                    
                    if actual_df is not None and len(actual_df) > 0:
                        first_row = actual_df.iloc[0]
                        actual_kwh = first_row.get("actual_heating_kwh")
                        if actual_kwh is not None and not pd.isna(actual_kwh):
                            has_actual = True
                            delta_kwh = predicted_kwh - actual_kwh
                            if actual_kwh > 0.01:
                                delta_pct = (delta_kwh / actual_kwh) * 100
                            total_actual += actual_kwh
                            compared_count += 1
                            abs_errors.append(abs(delta_kwh))
            except Exception as e:
                _Logger.debug("Could not get actual data for %s: %s", timestamp, e)
            
            comparison_results.append({
                "timestamp": timestamp,
                "predicted_kwh": round(predicted_kwh, 4),
                "actual_kwh": round(actual_kwh, 4) if actual_kwh is not None else None,
                "delta_kwh": round(delta_kwh, 4) if delta_kwh is not None else None,
                "delta_pct": round(delta_pct, 1) if delta_pct is not None else None,
                "has_actual": has_actual,
            })
        
        # Compute summary
        mae = sum(abs_errors) / len(abs_errors) if abs_errors else None
        mape = None
        if abs_errors and total_actual > 0:
            mape = (sum(abs_errors) / total_actual) * 100
        
        summary = {
            "total_predicted_kwh": round(total_predicted, 4),
            "total_actual_kwh": round(total_actual, 4) if compared_count > 0 else None,
            "slots_compared": compared_count,
            "slots_missing_actual": len(comparison_results) - compared_count,
            "mae_kwh": round(mae, 4) if mae is not None else None,
            "mape_pct": round(mape, 1) if mape is not None else None,
        }
        
        return jsonify({
            "status": "success",
            "prediction_id": prediction_id,
            "comparison": comparison_results,
            "summary": summary,
        })
        
    except Exception as e:
        _Logger.error("Error comparing prediction: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# SETTINGS OPTIMIZER ENDPOINTS
# =============================================================================

# Global optimizer state
_optimizer_progress: Optional[OptimizerProgress] = None
_optimizer_running: bool = False
_optimizer_thread: Optional[threading.Thread] = None


def _run_optimizer_in_thread():
    """Background thread function to run the optimizer."""
    global _optimizer_progress, _optimizer_running
    
    try:
        _Logger.info("Starting settings optimizer in background...")
        
        def progress_callback(progress):
            """Update global progress state."""
            global _optimizer_progress
            with _optimizer_lock:
                _optimizer_progress = progress
        
        # Run the optimization with parallel workers
        progress = run_optimization(
            train_single_step_fn=train_heating_demand_model,
            train_two_step_fn=train_two_step_heating_demand_model,
            build_dataset_fn=build_heating_feature_dataset,
            progress_callback=progress_callback,
            min_samples=50,
            max_workers=1,  # Reduced to 1 worker to prevent high RAM usage / OOM kills
            include_derived_features=True,  # Include derived features in optimization
        )
        
        with _optimizer_lock:
            _optimizer_progress = progress
        
        # Save results to database
        if progress:
            run_id = save_optimizer_run(progress)
            if run_id:
                _Logger.info("Saved optimizer run to database with ID: %d", run_id)
            else:
                _Logger.warning("Failed to save optimizer run to database")
        
    except Exception as e:
        _Logger.error("Error running optimizer: %s", e, exc_info=True)
        # Update progress with error
        with _optimizer_lock:
            if _optimizer_progress:
                _optimizer_progress.phase = "error"
                _optimizer_progress.error_message = str(e)
    
    finally:
        with _optimizer_lock:
            _optimizer_running = False


@app.post("/api/optimizer/run")
def run_optimizer():
    """
    Start the settings optimizer to find the best configuration.
    
    The optimizer will:
    1. Save current settings
    2. Cycle through different feature configurations
    3. Train both single-step and two-step models
    4. Find the configuration with the lowest Val MAPE (%)
    5. Save results to database
    
    This endpoint starts the optimization in the background and returns immediately.
    Use /api/optimizer/status to poll for progress.
    
    Response:
    {
        "status": "success",
        "message": "Optimizer started",
        "total_configurations": 12
    }
    """
    global _optimizer_progress, _optimizer_running, _optimizer_thread
    
    with _optimizer_lock:
        if _optimizer_running:
            return jsonify({
                "status": "error",
                "message": "Optimizer is already running",
            }), 400
        
        _optimizer_running = True
        _optimizer_progress = OptimizerProgress(
            total_configurations=0,
            completed_configurations=0,
            current_configuration="",
            current_model_type="",
            phase="initializing",
            start_time=datetime.now(),
        )
    
    try:
        # Start optimizer in background thread
        _optimizer_thread = threading.Thread(target=_run_optimizer_in_thread, daemon=True)
        _optimizer_thread.start()
        
        _Logger.info("Optimizer thread started")
        
        return jsonify({
            "status": "success",
            "message": "Optimizer started in background. Use /api/optimizer/status to check progress.",
            "running": True,
        })
        
    except Exception as e:
        with _optimizer_lock:
            _optimizer_running = False
        _Logger.error("Error starting optimizer: %s", e, exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@app.get("/api/optimizer/status")
def get_optimizer_status():
    """
    Get the current status of the optimizer.
    
    Response:
    {
        "status": "success",
        "running": false,
        "progress": {...}
    }
    """
    global _optimizer_progress, _optimizer_running
    
    with _optimizer_lock:
        if _optimizer_progress is None:
            return jsonify({
                "status": "success",
                "running": _optimizer_running,
                "progress": None,
                "message": "No optimization has been run yet",
            })
        
        progress = _optimizer_progress
        is_running = _optimizer_running
    
    response_data = {
        "status": "success",
        "running": is_running,
        "progress": {
            "phase": progress.phase,
            "total_configurations": progress.total_configurations,
            "completed_configurations": progress.completed_configurations,
            "current_configuration": progress.current_configuration,
            "current_model_type": progress.current_model_type,
            "log": progress.log_messages,
        },
    }
    
    # Include full results if optimization is complete
    if progress.phase in ["complete", "error"] and not is_running:
        # Try to load results from database (which have IDs)
        latest_run = get_latest_optimizer_run()
        if latest_run and latest_run.get("results"):
            response_data["progress"]["results"] = latest_run["results"]
        else:
            # Fallback to in-memory results (without IDs)
            response_data["progress"]["results"] = [
                {
                    "config_name": r.config_name,
                    "model_type": r.model_type,
                    "val_mape_pct": round(r.val_mape_pct, 2) if r.val_mape_pct is not None else None,
                    "val_mae_kwh": round(r.val_mae_kwh, 4) if r.val_mae_kwh is not None else None,
                    "val_r2": round(r.val_r2, 4) if r.val_r2 is not None else None,
                    "success": r.success,
                    "error_message": r.error_message,
                }
                for r in progress.results
            ]
        
        if progress.error_message:
            response_data["progress"]["error"] = progress.error_message
    
    if progress.best_result:
        response_data["progress"]["best_result"] = {
            "config_name": progress.best_result.config_name,
            "model_type": progress.best_result.model_type,
            "val_mape_pct": round(progress.best_result.val_mape_pct, 2) if progress.best_result.val_mape_pct else None,
            "experimental_features": progress.best_result.experimental_features,
        }
    
    return jsonify(response_data)


@app.post("/api/optimizer/apply")
def apply_optimizer_result():
    """
    Apply the best configuration found by the optimizer.
    
    This will save the experimental feature settings and optionally
    enable/disable two-step prediction based on the best result.
    
    Request body (optional):
    {
        "enable_two_step": true  // Whether to enable two-step if that was best
    }
    
    Response:
    {
        "status": "success",
        "message": "Best configuration applied",
        "applied_settings": {...}
    }
    """
    global _optimizer_progress
    
    if _optimizer_progress is None or _optimizer_progress.best_result is None:
        return jsonify({
            "status": "error",
            "message": "No optimization results to apply. Run the optimizer first.",
        }), 400
    
    try:
        data = request.get_json() or {}
        enable_two_step = data.get("enable_two_step", True)
        
        best = _optimizer_progress.best_result
        
        success = apply_best_configuration(
            best_result=best,
            enable_two_step=enable_two_step and best.model_type == "two_step",
        )
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Best configuration applied and saved",
                "applied_settings": {
                    "config_name": best.config_name,
                    "model_type": best.model_type,
                    "experimental_features": best.experimental_features,
                    "two_step_enabled": enable_two_step and best.model_type == "two_step",
                    "val_mape_pct": round(best.val_mape_pct, 2) if best.val_mape_pct else None,
                },
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to save configuration",
            }), 500
            
    except Exception as e:
        _Logger.error("Error applying optimizer result: %s", e)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@app.post("/api/optimizer/apply/<int:result_id>")
def apply_optimizer_result_by_id(result_id: int):
    """
    Apply a specific optimizer result by its database ID.
    
    This allows applying any result from the results table, not just the best one.
    
    Path parameter:
        result_id: The database ID of the optimizer result to apply
    
    Request body (optional):
    {
        "enable_two_step": true  // Whether to enable two-step if that was the model type
    }
    
    Response:
    {
        "status": "success",
        "message": "Configuration applied",
        "applied_settings": {...}
    }
    """
    try:
        data = request.get_json() or {}
        enable_two_step = data.get("enable_two_step", True)
        
        # Retrieve the result from database
        result_dict = get_optimizer_result_by_id(result_id)
        
        if not result_dict:
            return jsonify({
                "status": "error",
                "message": f"Optimizer result with ID {result_id} not found",
            }), 404
        
        # Convert to OptimizationResult for apply function
        from ml.optimizer import OptimizationResult
        
        result = OptimizationResult(
            config_name=result_dict["config_name"],
            model_type=result_dict["model_type"],
            experimental_features=result_dict["experimental_features"],
            val_mape_pct=result_dict["val_mape_pct"],
            val_mae_kwh=result_dict["val_mae_kwh"],
            val_r2=result_dict["val_r2"],
            train_samples=result_dict["train_samples"],
            val_samples=result_dict["val_samples"],
            success=result_dict["success"],
            error_message=result_dict["error_message"],
        )
        
        success = apply_best_configuration(
            best_result=result,
            enable_two_step=enable_two_step and result.model_type == "two_step",
        )
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Configuration applied and saved",
                "applied_settings": {
                    "config_name": result.config_name,
                    "model_type": result.model_type,
                    "experimental_features": result.experimental_features,
                    "two_step_enabled": enable_two_step and result.model_type == "two_step",
                    "val_mape_pct": round(result.val_mape_pct, 2) if result.val_mape_pct else None,
                },
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to save configuration",
            }), 500
            
    except Exception as e:
        _Logger.error("Error applying optimizer result %d: %s", result_id, e)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@app.get("/api/optimizer/runs")
def get_optimizer_runs():
    """
    Get a list of recent optimizer runs.
    
    Query parameters:
        limit: Maximum number of runs to return (default 10)
    
    Response:
    {
        "status": "success",
        "runs": [
            {
                "id": 1,
                "start_time": "2025-12-03T10:00:00",
                "end_time": "2025-12-03T10:15:00",
                "phase": "complete",
                "total_configurations": 12,
                "completed_configurations": 12,
                "best_result": {...}
            },
            ...
        ]
    }
    """
    try:
        limit = request.args.get("limit", 10, type=int)
        runs = list_optimizer_runs(limit=limit)
        
        return jsonify({
            "status": "success",
            "runs": runs,
        })
        
    except Exception as e:
        _Logger.error("Error retrieving optimizer runs: %s", e)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@app.get("/api/optimizer/runs/<int:run_id>")
def get_optimizer_run_details(run_id: int):
    """
    Get detailed information about a specific optimizer run.
    
    Path parameter:
        run_id: The database ID of the optimizer run
    
    Response:
    {
        "status": "success",
        "run": {
            "id": 1,
            "start_time": "2025-12-03T10:00:00",
            "end_time": "2025-12-03T10:15:00",
            "phase": "complete",
            "total_configurations": 12,
            "completed_configurations": 12,
            "best_result": {...},
            "results": [...]
        }
    }
    """
    try:
        run = get_optimizer_run_by_id(run_id)
        
        if not run:
            return jsonify({
                "status": "error",
                "message": f"Optimizer run with ID {run_id} not found",
            }), 404
        
        return jsonify({
            "status": "success",
            "run": run,
        })
        
    except Exception as e:
        _Logger.error("Error retrieving optimizer run %d: %s", run_id, e)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@app.get("/api/optimizer/latest")
def get_latest_optimizer_run_api():
    """
    Get the most recent optimizer run with all its results.
    
    Response:
    {
        "status": "success",
        "run": {
            "id": 1,
            "start_time": "2025-12-03T10:00:00",
            "end_time": "2025-12-03T10:15:00",
            "phase": "complete",
            "total_configurations": 12,
            "completed_configurations": 12,
            "best_result": {...},
            "results": [...]
        }
    }
    """
    try:
        run = get_latest_optimizer_run()
        
        if not run:
            return jsonify({
                "status": "success",
                "run": None,
                "message": "No optimizer runs found in database",
            })
        
        return jsonify({
            "status": "success",
            "run": run,
        })
        
    except Exception as e:
        _Logger.error("Error retrieving latest optimizer run: %s", e)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


# =============================================================================
# SENSOR CONFIGURATION ENDPOINTS
# =============================================================================


@app.get("/api/sensors/category_config")
def get_sensor_category_config_api():
    """
    Get the current sensor category configuration.
    
    Returns all sensors grouped by type with their configuration and status.
    Core sensors are always enabled and cannot be disabled.
    Experimental sensors can be enabled/disabled.
    
    Response:
    {
        "status": "success",
        "config": {
            "core_sensor_count": 6,
            "experimental_sensor_count": 5,
            "enabled_sensor_count": 6,
            "migrated_from_config_yaml": true
        },
        "sensors_by_type": {
            "weather": [...],
            "indoor": [...],
            "heating": [...],
            "usage": [...]
        },
        "enabled_entity_ids": ["sensor.outdoor_temp", ...]
    }
    """
    try:
        config = get_sensor_category_config()
        
        return jsonify({
            "status": "success",
            "config": {
                "core_sensor_count": len(CORE_SENSORS),
                "experimental_sensor_count": len(EXPERIMENTAL_SENSORS),
                "enabled_sensor_count": len(config.get_enabled_sensors()),
                "migrated_from_config_yaml": config.migrated_from_config_yaml,
            },
            "sensors_by_type": config.get_sensors_by_type(),
            "enabled_entity_ids": config.get_enabled_sensor_entity_ids(),
        })
    except Exception as e:
        _Logger.error("Error getting sensor category config: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/sensors/toggle")
def toggle_sensor():
    """
    Enable or disable an experimental sensor.
    
    Core sensors cannot be toggled (they are always enabled).
    
    Request body:
    {
        "category_name": "pressure",
        "enabled": true
    }
    
    Response:
    {
        "status": "success",
        "message": "Sensor 'pressure' is now enabled",
        "enabled_sensors": [...]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        category_name = data.get("category_name")
        enabled = data.get("enabled")
        
        if not category_name:
            return jsonify({
                "status": "error",
                "message": "category_name is required",
            }), 400
        
        if enabled is None:
            return jsonify({
                "status": "error",
                "message": "enabled is required (true or false)",
            }), 400
        
        # Verify sensor exists
        sensor_def = get_sensor_definition(category_name)
        if sensor_def is None:
            return jsonify({
                "status": "error",
                "message": f"Unknown sensor category: {category_name}",
            }), 400
        
        # Check if it's a core sensor
        if sensor_def.is_core:
            return jsonify({
                "status": "error",
                "message": f"Cannot toggle core sensor '{category_name}'. Core sensors are always enabled.",
            }), 400
        
        config = get_sensor_category_config()
        
        if enabled:
            result = config.enable_sensor(category_name)
        else:
            result = config.disable_sensor(category_name)
        
        if not result:
            return jsonify({
                "status": "error",
                "message": f"Failed to toggle sensor '{category_name}'",
            }), 500
        
        # Save configuration
        config.save()
        
        status = "enabled" if enabled else "disabled"
        return jsonify({
            "status": "success",
            "message": f"Sensor '{category_name}' is now {status}",
            "enabled_sensors": [s.category_name for s in config.get_enabled_sensors()],
        })
    except Exception as e:
        _Logger.error("Error toggling sensor: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/sensors/set_entity")
def set_sensor_entity():
    """
    Set the entity ID for a sensor category.
    
    Works for both core and experimental sensors.
    
    Request body:
    {
        "category_name": "outdoor_temp",
        "entity_id": "sensor.my_outdoor_temperature"
    }
    
    Response:
    {
        "status": "success",
        "message": "Entity ID for 'outdoor_temp' set to 'sensor.my_outdoor_temperature'",
        "sensor": {...}
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        category_name = data.get("category_name")
        entity_id = data.get("entity_id", "").strip()
        
        if not category_name:
            return jsonify({
                "status": "error",
                "message": "category_name is required",
            }), 400
        
        if not entity_id:
            return jsonify({
                "status": "error",
                "message": "entity_id is required and cannot be empty",
            }), 400
        
        # Verify sensor exists
        sensor_def = get_sensor_definition(category_name)
        if sensor_def is None:
            return jsonify({
                "status": "error",
                "message": f"Unknown sensor category: {category_name}",
            }), 400
        
        config = get_sensor_category_config()
        result = config.set_entity_id(category_name, entity_id)
        
        if not result:
            return jsonify({
                "status": "error",
                "message": f"Failed to set entity ID for '{category_name}'",
            }), 500
        
        # Save configuration
        config.save()
        
        sensor_config = config.get_sensor_config(category_name)
        
        return jsonify({
            "status": "success",
            "message": f"Entity ID for '{category_name}' set to '{entity_id}'",
            "sensor": {
                "category_name": category_name,
                "display_name": sensor_def.display_name,
                "entity_id": sensor_config.entity_id,
                "enabled": sensor_config.enabled,
                "is_core": sensor_def.is_core,
            },
        })
    except Exception as e:
        _Logger.error("Error setting sensor entity ID: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/sensors/set_unit")
def set_sensor_unit():
    """
    Set the unit for a sensor category.
    
    Works for both core and experimental sensors.
    
    Request body:
    {
        "category_name": "outdoor_temp",
        "unit": "°C"
    }
    
    Response:
    {
        "status": "success",
        "message": "Unit for 'outdoor_temp' set to '°C'",
        "sensor": {...}
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        category_name = data.get("category_name")
        unit = data.get("unit", "").strip()
        
        if not category_name:
            return jsonify({
                "status": "error",
                "message": "category_name is required",
            }), 400
        
        # Unit can be empty to clear/reset it
        
        # Verify sensor exists
        sensor_def = get_sensor_definition(category_name)
        if sensor_def is None:
            return jsonify({
                "status": "error",
                "message": f"Unknown sensor category: {category_name}",
            }), 400
        
        config = get_sensor_category_config()
        result = config.set_unit(category_name, unit)
        
        if not result:
            return jsonify({
                "status": "error",
                "message": f"Failed to set unit for '{category_name}'",
            }), 500
        
        # Save configuration
        config.save()
        
        sensor_config = config.get_sensor_config(category_name)
        
        return jsonify({
            "status": "success",
            "message": f"Unit for '{category_name}' set to '{unit}'" if unit else f"Unit for '{category_name}' cleared",
            "sensor": {
                "category_name": category_name,
                "display_name": sensor_def.display_name,
                "entity_id": sensor_config.entity_id,
                "unit": sensor_config.unit,
                "enabled": sensor_config.enabled,
                "is_core": sensor_def.is_core,
            },
        })
    except Exception as e:
        _Logger.error("Error setting sensor unit: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/sensors/definitions")
def get_sensor_definitions_api():
    """
    Get all sensor definitions (metadata).
    
    Returns the complete list of available sensors with their metadata,
    including core/experimental status, description, and units.
    
    Response:
    {
        "status": "success",
        "core_sensors": [...],
        "experimental_sensors": [...],
        "total_count": 11
    }
    """
    try:
        core_list = [s.to_dict() for s in CORE_SENSORS]
        experimental_list = [s.to_dict() for s in EXPERIMENTAL_SENSORS]
        
        return jsonify({
            "status": "success",
            "core_sensors": core_list,
            "experimental_sensors": experimental_list,
            "total_count": len(core_list) + len(experimental_list),
        })
    except Exception as e:
        _Logger.error("Error getting sensor definitions: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# VIRTUAL SENSORS ENDPOINTS
# =============================================================================


@app.get("/api/virtual_sensors/list")
def list_virtual_sensors():
    """
    Get list of all virtual sensors.
    
    Response:
    {
        "status": "success",
        "sensors": [...],
        "count": 5
    }
    """
    try:
        config = get_virtual_sensors_config()
        sensors = config.get_all_sensors()
        
        return jsonify({
            "status": "success",
            "sensors": [s.to_dict() for s in sensors],
            "count": len(sensors),
        })
    except Exception as e:
        _Logger.error("Error listing virtual sensors: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/virtual_sensors/add")
def add_virtual_sensor():
    """
    Add a new virtual sensor.
    
    Request body:
    {
        "name": "temp_delta",
        "display_name": "Temperature Delta",
        "description": "Difference between target and indoor temperature",
        "source_sensor1": "target_temp",
        "source_sensor2": "indoor_temp",
        "operation": "subtract",
        "unit": "°C"
    }
    
    Response:
    {
        "status": "success",
        "message": "Virtual sensor 'temp_delta' created",
        "sensor": {...}
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        # Validate required fields
        required_fields = ["name", "display_name", "description", "source_sensor1", "source_sensor2", "operation"]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "status": "error",
                    "message": f"'{field}' is required",
                }), 400
        
        # Validate operation
        try:
            operation = VirtualSensorOperation(data["operation"])
        except ValueError:
            return jsonify({
                "status": "error",
                "message": f"Invalid operation. Must be one of: {[op.value for op in VirtualSensorOperation]}",
            }), 400
        
        # Create virtual sensor definition
        sensor = VirtualSensorDefinition(
            name=data["name"],
            display_name=data["display_name"],
            description=data["description"],
            source_sensor1=data["source_sensor1"],
            source_sensor2=data["source_sensor2"],
            operation=operation,
            unit=data.get("unit", ""),
            enabled=data.get("enabled", True),
        )
        
        # Add to configuration
        config = get_virtual_sensors_config()
        if not config.add_sensor(sensor):
            return jsonify({
                "status": "error",
                "message": f"Virtual sensor '{sensor.name}' already exists",
            }), 400
        
        # Save configuration
        config.save()
        
        return jsonify({
            "status": "success",
            "message": f"Virtual sensor '{sensor.name}' created",
            "sensor": sensor.to_dict(),
        })
    except Exception as e:
        _Logger.error("Error adding virtual sensor: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.delete("/api/virtual_sensors/<name>")
def delete_virtual_sensor(name: str):
    """
    Delete a virtual sensor.
    
    Response:
    {
        "status": "success",
        "message": "Virtual sensor deleted"
    }
    """
    try:
        config = get_virtual_sensors_config()
        
        if not config.remove_sensor(name):
            return jsonify({
                "status": "error",
                "message": f"Virtual sensor '{name}' not found",
            }), 404
        
        # Save configuration
        config.save()
        
        return jsonify({
            "status": "success",
            "message": "Virtual sensor deleted",
        })
    except Exception as e:
        _Logger.error("Error deleting virtual sensor: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/virtual_sensors/<name>/toggle")
def toggle_virtual_sensor(name: str):
    """
    Enable or disable a virtual sensor.
    
    Request body:
    {
        "enabled": true
    }
    
    Response:
    {
        "status": "success",
        "message": "Virtual sensor enabled",
        "sensor": {...}
    }
    """
    try:
        data = request.get_json()
        
        if not data or "enabled" not in data:
            return jsonify({
                "status": "error",
                "message": "enabled field is required",
            }), 400
        
        config = get_virtual_sensors_config()
        sensor = config.get_sensor(name)
        
        if sensor is None:
            return jsonify({
                "status": "error",
                "message": f"Virtual sensor '{name}' not found",
            }), 404
        
        enabled = bool(data["enabled"])
        
        if enabled:
            config.enable_sensor(name)
        else:
            config.disable_sensor(name)
        
        # Save configuration
        config.save()
        
        status = "enabled" if enabled else "disabled"
        return jsonify({
            "status": "success",
            "message": f"Virtual sensor {status}",
            "sensor": sensor.to_dict(),
        })
    except Exception as e:
        _Logger.error("Error toggling virtual sensor: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# FEATURE STATS CONFIGURATION ENDPOINTS
# =============================================================================


@app.get("/api/feature_stats/config")
def get_feature_stats_config_api():
    """
    Get feature statistics configuration for all sensors.
    
    Shows which time-based statistics (avg_1h, avg_6h, avg_24h, avg_7d)
    are enabled for each sensor.
    
    Response:
    {
        "status": "success",
        "sensors": {
            "outdoor_temp": {
                "enabled_stats": ["avg_1h", "avg_6h", "avg_24h"],
                "stat_categories": ["outdoor_temp_avg_1h", "outdoor_temp_avg_6h", "outdoor_temp_avg_24h"]
            },
            ...
        },
        "all_stat_types": ["avg_1h", "avg_6h", "avg_24h", "avg_7d"]
    }
    """
    try:
        stats_config = get_feature_stats_config()
        sensor_category_config = get_sensor_category_config()
        virtual_sensors_config = get_virtual_sensors_config()
        
        # Get all available sensors (raw + virtual)
        all_sensors = []
        
        # Add raw sensors
        for sensor_config in sensor_category_config.get_enabled_sensors():
            all_sensors.append({
                "name": sensor_config.category_name,
                "type": "raw",
                "enabled": sensor_config.enabled,
            })
        
        # Add virtual sensors
        for virtual_sensor in virtual_sensors_config.get_enabled_sensors():
            all_sensors.append({
                "name": virtual_sensor.name,
                "type": "virtual",
                "enabled": virtual_sensor.enabled,
            })
        
        # Build response with stats configuration for each sensor
        sensors_with_stats = {}
        for sensor in all_sensors:
            sensor_name = sensor["name"]
            enabled_stats = stats_config.get_enabled_stats_for_sensor(sensor_name)
            stat_categories = [
                stats_config.get_sensor_config(sensor_name).get_stat_category_name(stat)
                for stat in enabled_stats
            ]
            
            sensors_with_stats[sensor_name] = {
                "sensor_type": sensor["type"],
                "enabled_stats": [s.value for s in enabled_stats],
                "stat_categories": stat_categories,
            }
        
        return jsonify({
            "status": "success",
            "sensors": sensors_with_stats,
            "all_stat_types": [s.value for s in StatType],
        })
    except Exception as e:
        _Logger.error("Error getting feature stats config: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/feature_stats/set")
def set_feature_stat():
    """
    Enable or disable a specific statistic for a sensor.
    
    Request body:
    {
        "sensor_name": "outdoor_temp",
        "stat_type": "avg_1h",
        "enabled": true
    }
    
    Response:
    {
        "status": "success",
        "message": "Statistic 'avg_1h' for 'outdoor_temp' enabled",
        "enabled_stats": ["avg_1h", "avg_6h", "avg_24h"]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body required",
            }), 400
        
        sensor_name = data.get("sensor_name")
        stat_type_str = data.get("stat_type")
        enabled = data.get("enabled")
        
        if not sensor_name or not stat_type_str or enabled is None:
            return jsonify({
                "status": "error",
                "message": "sensor_name, stat_type, and enabled are required",
            }), 400
        
        # Validate stat_type
        try:
            stat_type = StatType(stat_type_str)
        except ValueError:
            return jsonify({
                "status": "error",
                "message": f"Invalid stat_type. Must be one of: {[s.value for s in StatType]}",
            }), 400
        
        # Update configuration
        config = get_feature_stats_config()
        config.set_stat_enabled(sensor_name, stat_type, bool(enabled))
        
        # Save configuration
        config.save()
        
        # Get updated stats for this sensor
        enabled_stats = config.get_enabled_stats_for_sensor(sensor_name)
        
        status = "enabled" if enabled else "disabled"
        return jsonify({
            "status": "success",
            "message": f"Statistic '{stat_type.value}' for '{sensor_name}' {status}",
            "enabled_stats": [s.value for s in enabled_stats],
        })
    except Exception as e:
        _Logger.error("Error setting feature stat: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # Initialize database schema and sensor mappings before starting workers
    init_db_schema()
    sync_sensor_mappings()
    
    # Try to load existing models
    _get_model()
    _get_two_step_model()
    
    start_sensor_logging_worker()
    app.run(host="0.0.0.0", port=8099, debug=False)
