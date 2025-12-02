"""
Heating demand prediction model.

This module provides:
- Model training using gradient boosting regression
- Model persistence (save/load)
- Single-slot and scenario-based predictions

The model predicts heating energy demand (kWh) for a given prediction horizon
based on exogenous variables (weather, setpoints, time) and historical aggregations.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

_Logger = logging.getLogger(__name__)

# Default model storage path
MODEL_DIR = os.environ.get("MODEL_DIR", "/data")
MODEL_FILENAME = "heating_demand_model.joblib"


@dataclass
class TrainingMetrics:
    """Metrics from model training."""
    train_samples: int
    val_samples: int
    train_mae: float
    val_mae: float
    val_mape: float
    val_r2: float
    features: list[str]
    model_path: str


class ModelNotAvailableError(Exception):
    """Raised when model is not available for predictions."""
    pass


class HeatingDemandModel:
    """
    Heating demand prediction model.
    
    This class wraps a trained sklearn model and provides methods for
    single-slot and scenario-based predictions.
    """
    
    def __init__(
        self,
        model: Any = None,
        feature_names: list[str] = None,
        training_timestamp: datetime = None,
    ):
        self.model = model
        self.feature_names = feature_names or []
        self.training_timestamp = training_timestamp
        self._is_loaded = model is not None
    
    @property
    def is_available(self) -> bool:
        """Check if model is available for predictions."""
        return self._is_loaded and self.model is not None
    
    def predict(self, features: dict[str, float]) -> float:
        """
        Predict heating demand for a single time slot.
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            Predicted heating energy demand in kWh
            
        Raises:
            ModelNotAvailableError: If model is not loaded
            ValueError: If required features are missing
        """
        if not self.is_available:
            raise ModelNotAvailableError("Model not loaded")
        
        # Build feature vector in correct order
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        X = np.array([[features[f] for f in self.feature_names]])
        prediction = self.model.predict(X)[0]
        
        # Ensure non-negative prediction
        return max(0.0, float(prediction))
    
    def predict_batch(self, features_list: list[dict[str, float]]) -> list[float]:
        """
        Predict heating demand for multiple time slots.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of predicted heating energy demand values in kWh
        """
        if not self.is_available:
            raise ModelNotAvailableError("Model not loaded")
        
        if not features_list:
            return []
        
        # Build feature matrix
        X = np.array([
            [feat[f] for f in self.feature_names]
            for feat in features_list
        ])
        
        predictions = self.model.predict(X)
        return [max(0.0, float(p)) for p in predictions]


def _get_model_path() -> Path:
    """Get the path to the model file."""
    return Path(MODEL_DIR) / MODEL_FILENAME


def train_heating_demand_model(
    df: pd.DataFrame,
    target_col: str = "target_heating_kwh_1h",
    train_ratio: float = 0.8,
) -> tuple[HeatingDemandModel, TrainingMetrics]:
    """
    Train a heating demand prediction model.
    
    Uses time-based train/validation split (no shuffling).
    
    Args:
        df: Feature dataset with target column
        target_col: Name of the target column
        train_ratio: Fraction of data to use for training (earliest data)
        
    Returns:
        Tuple of (trained model, training metrics)
    """
    # Identify feature columns (everything except target)
    feature_cols = [c for c in df.columns if c != target_col]
    
    _Logger.info("Training model with %d features: %s", len(feature_cols), feature_cols)
    
    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Time-based split (data should already be sorted by time)
    split_idx = int(len(X) * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    _Logger.info(
        "Train/val split: %d training samples, %d validation samples",
        len(X_train),
        len(X_val),
    )
    
    # Train gradient boosting model
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    
    _Logger.info("Fitting model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    # MAPE - handle zero values
    y_val_nonzero = y_val[y_val > 0.01]
    val_pred_nonzero = val_pred[y_val > 0.01]
    if len(y_val_nonzero) > 0:
        val_mape = mean_absolute_percentage_error(y_val_nonzero, val_pred_nonzero)
    else:
        val_mape = float("nan")
    
    val_r2 = r2_score(y_val, val_pred)
    
    _Logger.info(
        "Training complete. Train MAE: %.4f kWh, Val MAE: %.4f kWh, "
        "Val MAPE: %.2f%%, Val R²: %.4f",
        train_mae,
        val_mae,
        val_mape * 100 if not np.isnan(val_mape) else float("nan"),
        val_r2,
    )
    
    # Save model
    model_path = _get_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        "model": model,
        "feature_names": feature_cols,
        "training_timestamp": datetime.now(),
        "metrics": {
            "train_mae": train_mae,
            "val_mae": val_mae,
            "val_mape": val_mape,
            "val_r2": val_r2,
        },
    }
    
    joblib.dump(model_data, model_path)
    _Logger.info("Model saved to %s", model_path)
    
    # Create model wrapper
    heating_model = HeatingDemandModel(
        model=model,
        feature_names=feature_cols,
        training_timestamp=model_data["training_timestamp"],
    )
    
    metrics = TrainingMetrics(
        train_samples=len(X_train),
        val_samples=len(X_val),
        train_mae=train_mae,
        val_mae=val_mae,
        val_mape=val_mape,
        val_r2=val_r2,
        features=feature_cols,
        model_path=str(model_path),
    )
    
    return heating_model, metrics


def load_heating_demand_model() -> Optional[HeatingDemandModel]:
    """
    Load a trained heating demand model from disk.
    
    Returns:
        HeatingDemandModel if successful, None if model file doesn't exist
        
    Raises:
        Exception: If model file exists but cannot be loaded (corrupt file)
    """
    model_path = _get_model_path()
    
    if not model_path.exists():
        _Logger.info("No model file found at %s", model_path)
        return None
    
    try:
        _Logger.info("Loading model from %s", model_path)
        model_data = joblib.load(model_path)
        
        model = HeatingDemandModel(
            model=model_data["model"],
            feature_names=model_data["feature_names"],
            training_timestamp=model_data.get("training_timestamp"),
        )
        
        _Logger.info(
            "Model loaded successfully. Features: %s",
            model.feature_names,
        )
        return model
        
    except Exception as e:
        _Logger.error("Failed to load model from %s: %s", model_path, e)
        raise


def predict_single_slot(
    model: HeatingDemandModel,
    features: dict[str, float],
) -> float:
    """
    Predict heating demand for a single time slot.
    
    This is a convenience wrapper around HeatingDemandModel.predict().
    
    Args:
        model: Trained heating demand model
        features: Feature dictionary
        
    Returns:
        Predicted heating energy demand in kWh
    """
    return model.predict(features)


def predict_scenario(
    model: HeatingDemandModel,
    scenario_features: list[dict[str, float]],
    update_historical: bool = False,
) -> list[float]:
    """
    Predict heating demand for a scenario (multiple time slots).
    
    This function supports scenario-based predictions for "what-if" analysis.
    
    Args:
        model: Trained heating demand model
        scenario_features: List of feature dictionaries, one per time slot
        update_historical: If True, update historical kWh features based on
            earlier predictions in the scenario (for cumulative effects)
            
    Returns:
        List of predicted heating energy demand values (kWh per time slot)
        
    Example:
        # Predict next 24 hours with varying setpoints
        scenario = []
        for hour in range(24):
            features = {
                "outdoor_temp": forecast[hour]["temp"],
                "wind": forecast[hour]["wind"],
                "target_temp": 20.0 if 6 <= hour <= 22 else 18.0,
                # ... other features
            }
            scenario.append(features)
        
        predictions = predict_scenario(model, scenario)
        total_kwh = sum(predictions)
    """
    if not model.is_available:
        raise ModelNotAvailableError("Model not loaded")
    
    if not scenario_features:
        return []
    
    if not update_historical:
        # Simple batch prediction
        return model.predict_batch(scenario_features)
    
    # Sequential prediction with historical updates
    predictions = []
    cumulative_kwh_6h = 0.0
    cumulative_kwh_24h = 0.0
    
    for i, features in enumerate(scenario_features):
        # Update historical kWh features if present
        updated_features = features.copy()
        
        if "heating_kwh_last_6h" in model.feature_names:
            # Add predictions from last 6 hours (hourly slots for scenario predictions)
            recent_preds = predictions[-6:] if len(predictions) >= 6 else predictions
            updated_features["heating_kwh_last_6h"] = (
                features.get("heating_kwh_last_6h", 0) + sum(recent_preds)
            )
        
        if "heating_kwh_last_24h" in model.feature_names:
            # Add predictions from last 24 hours (hourly slots for scenario predictions)
            recent_preds = predictions[-24:] if len(predictions) >= 24 else predictions
            updated_features["heating_kwh_last_24h"] = (
                features.get("heating_kwh_last_24h", 0) + sum(recent_preds)
            )
        
        pred = model.predict(updated_features)
        predictions.append(pred)
    
    return predictions
