"""
Prediction storage for storing and comparing predictions with actual data.

This module provides functionality to:
- Store predictions with their input parameters
- Compare stored predictions with actual sensor data
- Retrieve stored predictions for review

Predictions are stored in a JSON file at /data/stored_predictions.json.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

_Logger = logging.getLogger(__name__)

# Configuration file path for persistent prediction storage
# In Home Assistant add-ons, /data is the persistent data directory
PREDICTIONS_FILE_PATH = Path(os.environ.get("DATA_DIR", "/data")) / "stored_predictions.json"

# Maximum number of predictions to store
MAX_STORED_PREDICTIONS = 100


@dataclass
class StoredPrediction:
    """A stored prediction with metadata."""

    id: str  # Unique identifier (timestamp-based)
    created_at: str  # ISO timestamp when stored
    source: str  # Source of weather data (e.g., "weerlive", "manual", "historical")
    location: str  # Location used for forecast
    timeslots: list[dict]  # Input timeslots
    predictions: list[dict]  # Prediction results
    total_kwh: float  # Total predicted kWh
    model_type: str  # "single_step" or "two_step"


@dataclass
class ComparisonResult:
    """Result of comparing a prediction with actual data."""

    prediction_id: str
    timestamp: str
    predicted_kwh: float
    actual_kwh: float | None
    delta_kwh: float | None
    delta_pct: float | None
    has_actual: bool


@dataclass
class PredictionComparison:
    """Full comparison of a stored prediction with actual data."""

    prediction: StoredPrediction
    comparisons: list[ComparisonResult]
    summary: dict


def _load_predictions() -> list[dict]:
    """Load stored predictions from file.

    Returns:
        List of stored prediction dictionaries.
    """
    try:
        if PREDICTIONS_FILE_PATH.exists():
            with open(PREDICTIONS_FILE_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, OSError) as e:
        _Logger.warning("Error loading stored predictions: %s", e)
    return []


def _save_predictions(predictions: list[dict]) -> bool:
    """Save predictions to file.

    Args:
        predictions: List of prediction dictionaries.

    Returns:
        True if saved successfully.
    """
    try:
        # Ensure parent directory exists
        PREDICTIONS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Limit the number of stored predictions
        if len(predictions) > MAX_STORED_PREDICTIONS:
            predictions = predictions[-MAX_STORED_PREDICTIONS:]

        with open(PREDICTIONS_FILE_PATH, "w") as f:
            json.dump(predictions, f, indent=2)

        return True
    except OSError as e:
        _Logger.error("Error saving stored predictions: %s", e)
        return False


def store_prediction(
    timeslots: list[dict],
    predictions: list[dict],
    total_kwh: float,
    source: str = "manual",
    location: str = "",
    model_type: str = "single_step",
) -> tuple[bool, str | None, str | None]:
    """Store a prediction for later comparison.

    Args:
        timeslots: Input timeslot data.
        predictions: Prediction results.
        total_kwh: Total predicted kWh.
        source: Source of weather data.
        location: Location used for forecast.
        model_type: Type of model used ("single_step" or "two_step").

    Returns:
        Tuple of (success, error_message, prediction_id).
    """
    try:
        # Generate unique ID based on timestamp
        now = datetime.now()
        prediction_id = now.strftime("%Y%m%d_%H%M%S_%f")

        stored = StoredPrediction(
            id=prediction_id,
            created_at=now.isoformat(),
            source=source,
            location=location,
            timeslots=timeslots,
            predictions=predictions,
            total_kwh=total_kwh,
            model_type=model_type,
        )

        # Load existing predictions
        all_predictions = _load_predictions()

        # Add new prediction
        all_predictions.append(asdict(stored))

        # Save
        if _save_predictions(all_predictions):
            _Logger.info("Stored prediction %s with %d timeslots", prediction_id, len(timeslots))
            return True, None, prediction_id

        return False, "Failed to save prediction", None

    except Exception as e:
        _Logger.error("Error storing prediction: %s", e)
        return False, str(e), None


def get_stored_predictions() -> list[StoredPrediction]:
    """Get all stored predictions.

    Returns:
        List of StoredPrediction objects, newest first.
    """
    predictions = []
    for data in reversed(_load_predictions()):
        try:
            predictions.append(StoredPrediction(**data))
        except (TypeError, KeyError) as e:
            _Logger.warning("Skipping invalid stored prediction: %s", e)
    return predictions


def get_prediction_by_id(prediction_id: str) -> StoredPrediction | None:
    """Get a stored prediction by ID.

    Args:
        prediction_id: The prediction ID.

    Returns:
        StoredPrediction if found, None otherwise.
    """
    for data in _load_predictions():
        if data.get("id") == prediction_id:
            try:
                return StoredPrediction(**data)
            except (TypeError, KeyError):
                return None
    return None


def delete_prediction(prediction_id: str) -> bool:
    """Delete a stored prediction.

    Args:
        prediction_id: The prediction ID to delete.

    Returns:
        True if deleted successfully.
    """
    predictions = _load_predictions()
    original_count = len(predictions)

    predictions = [p for p in predictions if p.get("id") != prediction_id]

    if len(predictions) < original_count:
        return _save_predictions(predictions)
    return False


def compare_prediction_with_actual(
    prediction_id: str,
    get_actual_data_func,
) -> PredictionComparison | None:
    """Compare a stored prediction with actual sensor data.

    Args:
        prediction_id: The prediction ID to compare.
        get_actual_data_func: Function to get actual data for a time range.
            Should have signature: (start_time: datetime, end_time: datetime) -> dict[str, float]
            Returns a dict mapping timestamp ISO string to actual kWh value.

    Returns:
        PredictionComparison if prediction found, None otherwise.
    """
    prediction = get_prediction_by_id(prediction_id)
    if not prediction:
        return None

    comparisons = []
    total_predicted = 0.0
    total_actual = 0.0
    compared_count = 0
    abs_errors = []

    # Parse prediction timestamps
    for pred_data in prediction.predictions:
        timestamp = pred_data.get("timestamp", "")
        predicted_kwh = pred_data.get("predicted_kwh", 0.0)
        total_predicted += predicted_kwh

        # Try to get actual data for this timestamp
        actual_kwh = None
        if get_actual_data_func:
            try:
                actual_data = get_actual_data_func(timestamp)
                actual_kwh = actual_data
            except Exception as e:
                _Logger.debug("Could not get actual data for %s: %s", timestamp, e)

        delta_kwh = None
        delta_pct = None
        has_actual = actual_kwh is not None

        if has_actual:
            delta_kwh = predicted_kwh - actual_kwh
            if actual_kwh > 0.01:
                delta_pct = (delta_kwh / actual_kwh) * 100
            total_actual += actual_kwh
            compared_count += 1
            abs_errors.append(abs(delta_kwh))

        comparisons.append(ComparisonResult(
            prediction_id=prediction_id,
            timestamp=timestamp,
            predicted_kwh=predicted_kwh,
            actual_kwh=actual_kwh,
            delta_kwh=delta_kwh,
            delta_pct=delta_pct,
            has_actual=has_actual,
        ))

    # Compute summary
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else None
    mape = None
    if abs_errors and total_actual > 0:
        mape = (sum(abs_errors) / total_actual) * 100

    summary = {
        "total_predicted_kwh": round(total_predicted, 4),
        "total_actual_kwh": round(total_actual, 4) if compared_count > 0 else None,
        "slots_compared": compared_count,
        "slots_missing_actual": len(comparisons) - compared_count,
        "mae_kwh": round(mae, 4) if mae is not None else None,
        "mape_pct": round(mape, 1) if mape is not None else None,
    }

    return PredictionComparison(
        prediction=prediction,
        comparisons=comparisons,
        summary=summary,
    )


def get_prediction_list_summary() -> list[dict]:
    """Get a summary list of stored predictions for display.

    Returns:
        List of dictionaries with prediction summary info.
    """
    summaries = []
    for pred in get_stored_predictions():
        # Get time range from predictions
        timestamps = [p.get("timestamp", "") for p in pred.predictions if p.get("timestamp")]
        start_time = timestamps[0] if timestamps else None
        end_time = timestamps[-1] if timestamps else None

        summaries.append({
            "id": pred.id,
            "created_at": pred.created_at,
            "source": pred.source,
            "location": pred.location,
            "total_kwh": round(pred.total_kwh, 4),
            "slots_count": len(pred.predictions),
            "model_type": pred.model_type,
            "start_time": start_time,
            "end_time": end_time,
        })

    return summaries
