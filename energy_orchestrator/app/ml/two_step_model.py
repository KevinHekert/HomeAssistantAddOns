"""
Two-step heat pump prediction model.

This module provides a two-step approach for predicting hourly heat pump energy consumption:
1. Step 1 (Classifier): Predict whether the hour will be "active" (heating on) or "inactive" (no heating)
2. Step 2 (Regressor): For active hours only, predict how many kWh will be used

Key features:
- Automatic threshold detection for distinguishing "active" vs "inactive" hours
- No manual configuration required - threshold is derived from training data
- Threshold is stored with the model and reused for predictions

This approach solves the problem of:
- Overestimation when the pump is off (predicting kWh when there should be 0)
- Underestimation during heavy heating (averaging with inactive hours)
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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
)

_Logger = logging.getLogger(__name__)

# Default model storage path
MODEL_DIR = os.environ.get("MODEL_DIR", "/data")
TWO_STEP_MODEL_FILENAME = "heating_demand_two_step_model.joblib"

# Default threshold percentile for determining "active" vs "inactive"
# Uses 5th percentile of non-zero values as the threshold
DEFAULT_THRESHOLD_PERCENTILE = 5

# Minimum threshold to avoid classifying noise as active
MIN_ACTIVITY_THRESHOLD = 0.01  # 0.01 kWh = 10 Wh


@dataclass
class TwoStepTrainingMetrics:
    """Metrics from two-step model training."""
    # Threshold computation
    computed_threshold_kwh: float
    active_samples: int
    inactive_samples: int
    
    # Classifier metrics
    classifier_accuracy: float
    classifier_precision: float
    classifier_recall: float
    classifier_f1: float
    
    # Regressor metrics (on active samples only)
    regressor_train_samples: int
    regressor_val_samples: int
    regressor_train_mae: float
    regressor_val_mae: float
    regressor_val_mape: float
    regressor_val_r2: float
    
    # Overall metrics
    features: list[str]
    model_path: str


class TwoStepModelNotAvailableError(Exception):
    """Raised when two-step model is not available for predictions."""
    pass


@dataclass
class TwoStepPrediction:
    """Result of a two-step prediction for a single time slot."""
    is_active: bool
    predicted_kwh: float
    classifier_probability: float  # Probability of being active


class TwoStepHeatingDemandModel:
    """
    Two-step heating demand prediction model.
    
    This class combines:
    1. A classifier that predicts whether heating will be active
    2. A regressor that predicts kWh consumption for active hours
    
    The activity threshold is automatically computed from training data.
    """
    
    def __init__(
        self,
        classifier: Any = None,
        regressor: Any = None,
        activity_threshold_kwh: float = MIN_ACTIVITY_THRESHOLD,
        feature_names: list[str] = None,
        training_timestamp: datetime = None,
    ):
        self.classifier = classifier
        self.regressor = regressor
        self.activity_threshold_kwh = activity_threshold_kwh
        self.feature_names = feature_names or []
        self.training_timestamp = training_timestamp
        self._is_loaded = classifier is not None and regressor is not None
    
    @property
    def is_available(self) -> bool:
        """Check if model is available for predictions."""
        return self._is_loaded and self.classifier is not None and self.regressor is not None
    
    def predict(self, features: dict[str, float]) -> TwoStepPrediction:
        """
        Predict heating demand for a single time slot using two-step approach.
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            TwoStepPrediction with is_active flag and predicted_kwh
            
        Raises:
            TwoStepModelNotAvailableError: If model is not loaded
            ValueError: If required features are missing
        """
        if not self.is_available:
            raise TwoStepModelNotAvailableError("Two-step model not loaded")
        
        # Build feature vector in correct order
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        X = np.array([[features[f] for f in self.feature_names]])
        
        # Step 1: Classify as active or inactive
        is_active = bool(self.classifier.predict(X)[0])
        classifier_prob = float(self.classifier.predict_proba(X)[0][1])
        
        # Step 2: If active, predict kWh; otherwise return 0
        if is_active:
            predicted_kwh = float(self.regressor.predict(X)[0])
            # Ensure non-negative
            predicted_kwh = max(0.0, predicted_kwh)
        else:
            predicted_kwh = 0.0
        
        return TwoStepPrediction(
            is_active=is_active,
            predicted_kwh=predicted_kwh,
            classifier_probability=classifier_prob,
        )
    
    def predict_batch(self, features_list: list[dict[str, float]]) -> list[TwoStepPrediction]:
        """
        Predict heating demand for multiple time slots.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of TwoStepPrediction objects
        """
        if not self.is_available:
            raise TwoStepModelNotAvailableError("Two-step model not loaded")
        
        if not features_list:
            return []
        
        # Build feature matrix
        X = np.array([
            [feat[f] for f in self.feature_names]
            for feat in features_list
        ])
        
        # Step 1: Classify all samples
        is_active_arr = self.classifier.predict(X)
        proba_arr = self.classifier.predict_proba(X)[:, 1]
        
        # Step 2: Predict kWh for all (we'll zero out inactive ones)
        predicted_kwh_arr = self.regressor.predict(X)
        
        results = []
        for i in range(len(features_list)):
            is_active = bool(is_active_arr[i])
            if is_active:
                kwh = max(0.0, float(predicted_kwh_arr[i]))
            else:
                kwh = 0.0
            
            results.append(TwoStepPrediction(
                is_active=is_active,
                predicted_kwh=kwh,
                classifier_probability=float(proba_arr[i]),
            ))
        
        return results
    
    def predict_kwh_values(self, features_list: list[dict[str, float]]) -> list[float]:
        """
        Predict heating demand values only (for compatibility with existing API).
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of predicted kWh values (0 for inactive hours)
        """
        predictions = self.predict_batch(features_list)
        return [p.predicted_kwh for p in predictions]


def _compute_activity_threshold(
    target_values: np.ndarray,
    percentile: int = DEFAULT_THRESHOLD_PERCENTILE,
) -> float:
    """
    Automatically compute the threshold for distinguishing active vs inactive hours.
    
    This function analyzes the distribution of target values (kWh) and determines
    a threshold that separates "noise/standby" from "real heating".
    
    The algorithm:
    1. Filter to non-zero positive values
    2. If fewer than 10 positive samples, use MIN_ACTIVITY_THRESHOLD
    3. Compute the specified percentile of positive values
    4. Use max(percentile_value, MIN_ACTIVITY_THRESHOLD)
    
    Args:
        target_values: Array of target heating kWh values
        percentile: Percentile to use for threshold computation (default 5)
        
    Returns:
        Computed threshold in kWh
    """
    # Filter to positive values
    positive_values = target_values[target_values > 0]
    
    if len(positive_values) < 10:
        _Logger.warning(
            "Fewer than 10 positive heating samples (%d), using minimum threshold",
            len(positive_values),
        )
        return MIN_ACTIVITY_THRESHOLD
    
    # Compute percentile of positive values
    percentile_threshold = float(np.percentile(positive_values, percentile))
    
    # Ensure threshold is at least the minimum
    threshold = max(percentile_threshold, MIN_ACTIVITY_THRESHOLD)
    
    _Logger.info(
        "Computed activity threshold: %.4f kWh (percentile=%d, min=%.4f, "
        "positive_samples=%d, total_samples=%d)",
        threshold,
        percentile,
        MIN_ACTIVITY_THRESHOLD,
        len(positive_values),
        len(target_values),
    )
    
    return threshold


def _get_two_step_model_path() -> Path:
    """Get the path to the two-step model file."""
    return Path(MODEL_DIR) / TWO_STEP_MODEL_FILENAME


def train_two_step_heating_demand_model(
    df: pd.DataFrame,
    target_col: str = "target_heating_kwh_1h",
    train_ratio: float = 0.8,
    threshold_percentile: int = DEFAULT_THRESHOLD_PERCENTILE,
) -> tuple[TwoStepHeatingDemandModel, TwoStepTrainingMetrics]:
    """
    Train a two-step heating demand prediction model.
    
    This function:
    1. Automatically computes the activity threshold from training data
    2. Creates binary labels for classifier training
    3. Trains a classifier to predict active/inactive
    4. Trains a regressor on active samples only
    5. Saves the combined model
    
    Args:
        df: Feature dataset with target column
        target_col: Name of the target column
        train_ratio: Fraction of data to use for training
        threshold_percentile: Percentile for threshold computation
        
    Returns:
        Tuple of (trained model, training metrics)
    """
    # Identify feature columns
    feature_cols = [c for c in df.columns if c != target_col]
    
    _Logger.info("Training two-step model with %d features: %s", len(feature_cols), feature_cols)
    
    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Step 1: Compute activity threshold automatically
    activity_threshold = _compute_activity_threshold(y, threshold_percentile)
    
    # Create binary labels: 1 = active (heating on), 0 = inactive
    y_binary = (y >= activity_threshold).astype(int)
    
    active_count = int(y_binary.sum())
    inactive_count = len(y_binary) - active_count
    
    _Logger.info(
        "Activity threshold: %.4f kWh. Active samples: %d (%.1f%%), Inactive samples: %d (%.1f%%)",
        activity_threshold,
        active_count,
        100 * active_count / len(y_binary),
        inactive_count,
        100 * inactive_count / len(y_binary),
    )
    
    # Time-based split
    split_idx = int(len(X) * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    y_binary_train, y_binary_val = y_binary[:split_idx], y_binary[split_idx:]
    
    _Logger.info(
        "Train/val split: %d training samples, %d validation samples",
        len(X_train),
        len(X_val),
    )
    
    # Step 2: Train classifier
    _Logger.info("Training classifier...")
    classifier = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    classifier.fit(X_train, y_binary_train)
    
    # Evaluate classifier
    y_binary_pred_train = classifier.predict(X_train)
    y_binary_pred_val = classifier.predict(X_val)
    
    classifier_accuracy = accuracy_score(y_binary_val, y_binary_pred_val)
    classifier_precision = precision_score(y_binary_val, y_binary_pred_val, zero_division=0)
    classifier_recall = recall_score(y_binary_val, y_binary_pred_val, zero_division=0)
    classifier_f1 = f1_score(y_binary_val, y_binary_pred_val, zero_division=0)
    
    _Logger.info(
        "Classifier metrics - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
        classifier_accuracy,
        classifier_precision,
        classifier_recall,
        classifier_f1,
    )
    
    # Step 3: Train regressor on ACTIVE samples only
    _Logger.info("Training regressor on active samples only...")
    
    # Filter to active samples for regressor training
    active_mask_train = y_binary_train == 1
    active_mask_val = y_binary_val == 1
    
    X_train_active = X_train[active_mask_train]
    y_train_active = y_train[active_mask_train]
    X_val_active = X_val[active_mask_val]
    y_val_active = y_val[active_mask_val]
    
    _Logger.info(
        "Regressor training: %d active training samples, %d active validation samples",
        len(X_train_active),
        len(X_val_active),
    )
    
    regressor = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    regressor.fit(X_train_active, y_train_active)
    
    # Evaluate regressor on active samples
    y_pred_train_active = regressor.predict(X_train_active)
    y_pred_val_active = regressor.predict(X_val_active)
    
    regressor_train_mae = mean_absolute_error(y_train_active, y_pred_train_active)
    regressor_val_mae = mean_absolute_error(y_val_active, y_pred_val_active)
    
    # MAPE - handle zero values
    y_val_nonzero = y_val_active[y_val_active > 0.01]
    y_pred_nonzero = y_pred_val_active[y_val_active > 0.01]
    if len(y_val_nonzero) > 0:
        regressor_val_mape = mean_absolute_percentage_error(y_val_nonzero, y_pred_nonzero)
    else:
        regressor_val_mape = float("nan")
    
    regressor_val_r2 = r2_score(y_val_active, y_pred_val_active)
    
    _Logger.info(
        "Regressor metrics (active samples) - Train MAE: %.4f kWh, Val MAE: %.4f kWh, "
        "Val MAPE: %.2f%%, Val R²: %.4f",
        regressor_train_mae,
        regressor_val_mae,
        regressor_val_mape * 100 if not np.isnan(regressor_val_mape) else float("nan"),
        regressor_val_r2,
    )
    
    # Step 4: Save combined model
    model_path = _get_two_step_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        "classifier": classifier,
        "regressor": regressor,
        "activity_threshold_kwh": activity_threshold,
        "feature_names": feature_cols,
        "training_timestamp": datetime.now(),
        "metrics": {
            "classifier_accuracy": classifier_accuracy,
            "classifier_precision": classifier_precision,
            "classifier_recall": classifier_recall,
            "classifier_f1": classifier_f1,
            "regressor_train_mae": regressor_train_mae,
            "regressor_val_mae": regressor_val_mae,
            "regressor_val_mape": regressor_val_mape,
            "regressor_val_r2": regressor_val_r2,
        },
    }
    
    joblib.dump(model_data, model_path)
    _Logger.info("Two-step model saved to %s", model_path)
    
    # Create model wrapper
    two_step_model = TwoStepHeatingDemandModel(
        classifier=classifier,
        regressor=regressor,
        activity_threshold_kwh=activity_threshold,
        feature_names=feature_cols,
        training_timestamp=model_data["training_timestamp"],
    )
    
    metrics = TwoStepTrainingMetrics(
        computed_threshold_kwh=activity_threshold,
        active_samples=active_count,
        inactive_samples=inactive_count,
        classifier_accuracy=classifier_accuracy,
        classifier_precision=classifier_precision,
        classifier_recall=classifier_recall,
        classifier_f1=classifier_f1,
        regressor_train_samples=len(X_train_active),
        regressor_val_samples=len(X_val_active),
        regressor_train_mae=regressor_train_mae,
        regressor_val_mae=regressor_val_mae,
        regressor_val_mape=regressor_val_mape,
        regressor_val_r2=regressor_val_r2,
        features=feature_cols,
        model_path=str(model_path),
    )
    
    return two_step_model, metrics


def load_two_step_heating_demand_model() -> Optional[TwoStepHeatingDemandModel]:
    """
    Load a trained two-step heating demand model from disk.
    
    Returns:
        TwoStepHeatingDemandModel if successful, None if model file doesn't exist
        
    Raises:
        Exception: If model file exists but cannot be loaded
    """
    model_path = _get_two_step_model_path()
    
    if not model_path.exists():
        _Logger.info("No two-step model file found at %s", model_path)
        return None
    
    try:
        _Logger.info("Loading two-step model from %s", model_path)
        model_data = joblib.load(model_path)
        
        model = TwoStepHeatingDemandModel(
            classifier=model_data["classifier"],
            regressor=model_data["regressor"],
            activity_threshold_kwh=model_data["activity_threshold_kwh"],
            feature_names=model_data["feature_names"],
            training_timestamp=model_data.get("training_timestamp"),
        )
        
        _Logger.info(
            "Two-step model loaded successfully. Threshold: %.4f kWh, Features: %s",
            model.activity_threshold_kwh,
            model.feature_names,
        )
        return model
        
    except Exception as e:
        _Logger.error("Failed to load two-step model from %s: %s", model_path, e)
        raise


def predict_two_step_scenario(
    model: TwoStepHeatingDemandModel,
    scenario_features: list[dict[str, float]],
    update_historical: bool = False,
) -> list[TwoStepPrediction]:
    """
    Predict heating demand for a scenario using two-step approach.
    
    This function predicts for multiple time slots, returning both the
    active/inactive classification and the predicted kWh for each slot.
    
    Args:
        model: Trained two-step heating demand model
        scenario_features: List of feature dictionaries, one per time slot
        update_historical: If True, update historical kWh features based on
            earlier predictions in the scenario (for cumulative effects)
            
    Returns:
        List of TwoStepPrediction objects
        
    Example:
        predictions = predict_two_step_scenario(model, scenario)
        for pred in predictions:
            if pred.is_active:
                print(f"Active hour: {pred.predicted_kwh:.2f} kWh")
            else:
                print("Inactive hour: 0 kWh")
    """
    if not model.is_available:
        raise TwoStepModelNotAvailableError("Two-step model not loaded")
    
    if not scenario_features:
        return []
    
    if not update_historical:
        # Simple batch prediction
        return model.predict_batch(scenario_features)
    
    # Sequential prediction with historical updates
    predictions = []
    
    for i, features in enumerate(scenario_features):
        # Update historical kWh features if present
        updated_features = features.copy()
        
        if "heating_kwh_last_6h" in model.feature_names:
            # Add predictions from last 6 hours
            recent_preds = [p.predicted_kwh for p in predictions[-72:]]
            updated_features["heating_kwh_last_6h"] = (
                features.get("heating_kwh_last_6h", 0) + sum(recent_preds)
            )
        
        if "heating_kwh_last_24h" in model.feature_names:
            # Add predictions from last 24 hours
            recent_preds = [p.predicted_kwh for p in predictions[-288:]]
            updated_features["heating_kwh_last_24h"] = (
                features.get("heating_kwh_last_24h", 0) + sum(recent_preds)
            )
        
        pred = model.predict(updated_features)
        predictions.append(pred)
    
    return predictions
