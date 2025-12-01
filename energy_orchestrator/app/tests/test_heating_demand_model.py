"""
Tests for the heating demand model module.

Tests:
1. Model training
2. Model persistence (save/load)
3. Single-slot predictions
4. Scenario-based predictions
5. Error handling
"""

import os
import tempfile
import pytest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.heating_demand_model import (
    HeatingDemandModel,
    ModelNotAvailableError,
    TrainingMetrics,
    train_heating_demand_model,
    load_heating_demand_model,
    predict_single_slot,
    predict_scenario,
    _get_model_path,
)
import ml.heating_demand_model as model_module


@pytest.fixture
def sample_dataset():
    """Create a sample training dataset."""
    np.random.seed(42)
    n_samples = 500
    
    # Generate realistic-looking data
    df = pd.DataFrame({
        "outdoor_temp": np.random.normal(10, 5, n_samples),
        "wind": np.random.exponential(3, n_samples),
        "humidity": np.random.uniform(50, 95, n_samples),
        "indoor_temp": np.random.normal(20, 1, n_samples),
        "target_temp": np.random.choice([18, 20, 21], n_samples),
        "outdoor_temp_avg_1h": np.random.normal(10, 4, n_samples),
        "heating_kwh_last_6h": np.random.exponential(2, n_samples),
        "hour_of_day": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "is_weekend": np.random.choice([0, 1], n_samples),
        "is_night": np.random.choice([0, 1], n_samples),
    })
    
    # Target: correlated with temperature difference and historical usage
    df["target_heating_kwh_1h"] = (
        0.3 * (df["target_temp"] - df["outdoor_temp"]).clip(lower=0) +
        0.1 * df["heating_kwh_last_6h"] +
        np.random.normal(0, 0.2, n_samples)
    ).clip(lower=0)
    
    return df


@pytest.fixture
def temp_model_dir(tmp_path, monkeypatch):
    """Create a temporary directory for model storage."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(model_module, "MODEL_DIR", str(model_dir))
    return model_dir


class TestHeatingDemandModel:
    """Test the HeatingDemandModel class."""
    
    def test_model_not_available_when_empty(self):
        """Model is not available when not loaded."""
        model = HeatingDemandModel()
        assert not model.is_available
    
    def test_model_available_when_loaded(self):
        """Model is available when properly initialized."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        sklearn_model = GradientBoostingRegressor(n_estimators=10)
        sklearn_model.fit([[0, 1], [1, 0]], [0, 1])
        
        model = HeatingDemandModel(
            model=sklearn_model,
            feature_names=["feat1", "feat2"],
        )
        
        assert model.is_available
    
    def test_predict_raises_when_not_available(self):
        """Predict raises error when model not available."""
        model = HeatingDemandModel()
        
        with pytest.raises(ModelNotAvailableError):
            model.predict({"feat1": 1.0})
    
    def test_predict_raises_on_missing_features(self):
        """Predict raises error when features are missing."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        sklearn_model = GradientBoostingRegressor(n_estimators=10)
        sklearn_model.fit([[0, 1], [1, 0]], [0, 1])
        
        model = HeatingDemandModel(
            model=sklearn_model,
            feature_names=["feat1", "feat2"],
        )
        
        with pytest.raises(ValueError) as exc_info:
            model.predict({"feat1": 1.0})  # Missing feat2
        
        assert "Missing features" in str(exc_info.value)
    
    def test_predict_returns_non_negative(self):
        """Predictions are always non-negative."""
        from sklearn.linear_model import LinearRegression
        
        # Train a model that might predict negative values
        sklearn_model = LinearRegression()
        sklearn_model.fit([[10], [20]], [0.5, 1.5])
        
        model = HeatingDemandModel(
            model=sklearn_model,
            feature_names=["temp"],
        )
        
        # Predict with a value that might give negative
        result = model.predict({"temp": -100})
        
        assert result >= 0


class TestTrainHeatingDemandModel:
    """Test the train_heating_demand_model function."""
    
    def test_trains_successfully(self, sample_dataset, temp_model_dir):
        """Model trains successfully with valid data."""
        model, metrics = train_heating_demand_model(sample_dataset)
        
        assert model.is_available
        assert metrics.train_samples > 0
        assert metrics.val_samples > 0
        assert metrics.train_mae >= 0
        assert metrics.val_mae >= 0
        assert len(metrics.features) > 0
    
    def test_saves_model_to_disk(self, sample_dataset, temp_model_dir):
        """Model is saved to disk after training."""
        model, metrics = train_heating_demand_model(sample_dataset)
        
        model_path = temp_model_dir / "heating_demand_model.joblib"
        assert model_path.exists()
    
    def test_time_based_split(self, sample_dataset, temp_model_dir):
        """Uses time-based train/val split (earliest for training)."""
        model, metrics = train_heating_demand_model(
            sample_dataset,
            train_ratio=0.7,
        )
        
        # With 500 samples and 0.7 ratio, expect ~350 train, ~150 val
        assert 300 <= metrics.train_samples <= 400
        assert 100 <= metrics.val_samples <= 200
    
    def test_metrics_are_reasonable(self, sample_dataset, temp_model_dir):
        """Training metrics are within reasonable bounds."""
        model, metrics = train_heating_demand_model(sample_dataset)
        
        # MAE should be positive and not too large
        assert 0 < metrics.val_mae < 10
        
        # R² can be negative but should be reasonable
        assert metrics.val_r2 > -1


class TestLoadHeatingDemandModel:
    """Test the load_heating_demand_model function."""
    
    def test_returns_none_when_no_model(self, temp_model_dir):
        """Returns None when no model file exists."""
        model = load_heating_demand_model()
        assert model is None
    
    def test_loads_trained_model(self, sample_dataset, temp_model_dir):
        """Loads a previously trained model."""
        # Train and save
        trained_model, _ = train_heating_demand_model(sample_dataset)
        
        # Load
        loaded_model = load_heating_demand_model()
        
        assert loaded_model is not None
        assert loaded_model.is_available
        assert loaded_model.feature_names == trained_model.feature_names
    
    def test_raises_on_corrupt_file(self, temp_model_dir):
        """Raises exception when model file is corrupt."""
        model_path = temp_model_dir / "heating_demand_model.joblib"
        model_path.write_text("not a valid joblib file")
        
        with pytest.raises(Exception):
            load_heating_demand_model()


class TestPredictSingleSlot:
    """Test the predict_single_slot function."""
    
    def test_predicts_correctly(self, sample_dataset, temp_model_dir):
        """Makes predictions for single slot."""
        model, _ = train_heating_demand_model(sample_dataset)
        
        features = {
            "outdoor_temp": 5.0,
            "wind": 3.0,
            "humidity": 75.0,
            "indoor_temp": 20.0,
            "target_temp": 21.0,
            "outdoor_temp_avg_1h": 5.0,
            "heating_kwh_last_6h": 2.0,
            "hour_of_day": 14,
            "day_of_week": 0,
            "is_weekend": 0,
            "is_night": 0,
        }
        
        prediction = predict_single_slot(model, features)
        
        assert isinstance(prediction, float)
        assert prediction >= 0


class TestPredictScenario:
    """Test the predict_scenario function."""
    
    def test_predicts_multiple_slots(self, sample_dataset, temp_model_dir):
        """Predicts for multiple time slots."""
        model, _ = train_heating_demand_model(sample_dataset)
        
        base_features = {
            "outdoor_temp": 5.0,
            "wind": 3.0,
            "humidity": 75.0,
            "indoor_temp": 20.0,
            "target_temp": 21.0,
            "outdoor_temp_avg_1h": 5.0,
            "heating_kwh_last_6h": 2.0,
            "hour_of_day": 14,
            "day_of_week": 0,
            "is_weekend": 0,
            "is_night": 0,
        }
        
        # Create scenario with 24 time slots
        scenario = []
        for hour in range(24):
            features = base_features.copy()
            features["hour_of_day"] = hour
            features["is_night"] = 1 if hour >= 23 or hour < 7 else 0
            scenario.append(features)
        
        predictions = predict_scenario(model, scenario)
        
        assert len(predictions) == 24
        assert all(p >= 0 for p in predictions)
    
    def test_empty_scenario(self, sample_dataset, temp_model_dir):
        """Handles empty scenario gracefully."""
        model, _ = train_heating_demand_model(sample_dataset)
        
        predictions = predict_scenario(model, [])
        
        assert predictions == []
    
    def test_scenario_with_varying_setpoints(self, sample_dataset, temp_model_dir):
        """Handles scenarios with varying setpoints."""
        model, _ = train_heating_demand_model(sample_dataset)
        
        base_features = {
            "outdoor_temp": 5.0,
            "wind": 3.0,
            "humidity": 75.0,
            "indoor_temp": 20.0,
            "target_temp": 20.0,
            "outdoor_temp_avg_1h": 5.0,
            "heating_kwh_last_6h": 2.0,
            "hour_of_day": 12,
            "day_of_week": 0,
            "is_weekend": 0,
            "is_night": 0,
        }
        
        # Low setpoint scenario
        low_scenario = [base_features.copy() for _ in range(5)]
        for f in low_scenario:
            f["target_temp"] = 18.0
        
        # High setpoint scenario
        high_scenario = [base_features.copy() for _ in range(5)]
        for f in high_scenario:
            f["target_temp"] = 22.0
        
        low_predictions = predict_scenario(model, low_scenario)
        high_predictions = predict_scenario(model, high_scenario)
        
        # Higher setpoint should generally need more energy
        # (this depends on model training, but is a reasonable expectation)
        assert sum(high_predictions) >= sum(low_predictions) * 0.8
    
    def test_raises_when_model_not_available(self):
        """Raises error when model not available."""
        model = HeatingDemandModel()
        
        with pytest.raises(ModelNotAvailableError):
            predict_scenario(model, [{"feat": 1.0}])


class TestModelIntegration:
    """Integration tests for the complete model workflow."""
    
    def test_train_save_load_predict_workflow(self, sample_dataset, temp_model_dir):
        """Tests complete train -> save -> load -> predict workflow."""
        # Train
        original_model, metrics = train_heating_demand_model(sample_dataset)
        
        # Verify saved
        model_path = temp_model_dir / "heating_demand_model.joblib"
        assert model_path.exists()
        
        # Load
        loaded_model = load_heating_demand_model()
        
        # Predict with both models
        features = {
            "outdoor_temp": 8.0,
            "wind": 2.0,
            "humidity": 70.0,
            "indoor_temp": 19.5,
            "target_temp": 20.0,
            "outdoor_temp_avg_1h": 7.5,
            "heating_kwh_last_6h": 1.5,
            "hour_of_day": 10,
            "day_of_week": 2,
            "is_weekend": 0,
            "is_night": 0,
        }
        
        pred_original = original_model.predict(features)
        pred_loaded = loaded_model.predict(features)
        
        # Should produce identical results
        assert abs(pred_original - pred_loaded) < 0.0001
    
    def test_model_retraining(self, sample_dataset, temp_model_dir):
        """Model can be retrained and overwrites previous model."""
        # Train first model
        model1, metrics1 = train_heating_demand_model(sample_dataset)
        
        # Modify dataset slightly
        modified_dataset = sample_dataset.copy()
        modified_dataset["target_heating_kwh_1h"] *= 1.5
        
        # Retrain
        model2, metrics2 = train_heating_demand_model(modified_dataset)
        
        # Load should return the second model
        loaded = load_heating_demand_model()
        
        # The loaded model should match the retrained one
        features = {
            "outdoor_temp": 8.0,
            "wind": 2.0,
            "humidity": 70.0,
            "indoor_temp": 19.5,
            "target_temp": 20.0,
            "outdoor_temp_avg_1h": 7.5,
            "heating_kwh_last_6h": 1.5,
            "hour_of_day": 10,
            "day_of_week": 2,
            "is_weekend": 0,
            "is_night": 0,
        }
        
        pred2 = model2.predict(features)
        pred_loaded = loaded.predict(features)
        
        assert abs(pred2 - pred_loaded) < 0.0001
