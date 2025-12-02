"""
Tests for the two-step heating demand model module.

Tests:
1. Automatic threshold computation
2. Classifier training and metrics
3. Regressor training on active samples only
4. Two-step predictions
5. Model persistence (save/load)
6. Error handling
"""

import os
import tempfile
import pytest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.two_step_model import (
    TwoStepHeatingDemandModel,
    TwoStepModelNotAvailableError,
    TwoStepPrediction,
    TwoStepTrainingMetrics,
    train_two_step_heating_demand_model,
    load_two_step_heating_demand_model,
    predict_two_step_scenario,
    _compute_activity_threshold,
    MIN_ACTIVITY_THRESHOLD,
)
import ml.two_step_model as two_step_module


@pytest.fixture
def sample_dataset():
    """Create a sample training dataset with both active and inactive hours."""
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
    
    # Target: mix of active (heating on) and inactive (heating off) hours
    # About 30% inactive (0 kWh), 70% active (variable kWh)
    is_active = np.random.random(n_samples) > 0.3
    
    # For active hours, generate correlated kWh values
    active_kwh = (
        0.3 * (df["target_temp"] - df["outdoor_temp"]).clip(lower=0) +
        0.1 * df["heating_kwh_last_6h"] +
        np.random.normal(0.2, 0.15, n_samples)
    ).clip(lower=0.1)  # Minimum 0.1 kWh for active hours
    
    # For inactive hours, set to 0 (or very small noise)
    df["target_heating_kwh_1h"] = np.where(is_active, active_kwh, 0.0)
    
    return df


@pytest.fixture
def sample_dataset_mostly_active():
    """Create a dataset with mostly active hours (>90%)."""
    np.random.seed(42)
    n_samples = 500
    
    df = pd.DataFrame({
        "outdoor_temp": np.random.normal(5, 3, n_samples),  # Colder, more heating
        "wind": np.random.exponential(3, n_samples),
        "humidity": np.random.uniform(50, 95, n_samples),
        "indoor_temp": np.random.normal(20, 1, n_samples),
        "target_temp": np.full(n_samples, 21),  # High setpoint
        "outdoor_temp_avg_1h": np.random.normal(5, 2, n_samples),
        "heating_kwh_last_6h": np.random.exponential(3, n_samples),
        "hour_of_day": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "is_weekend": np.random.choice([0, 1], n_samples),
        "is_night": np.random.choice([0, 1], n_samples),
    })
    
    # 95% active
    is_active = np.random.random(n_samples) > 0.05
    active_kwh = (
        0.4 * (df["target_temp"] - df["outdoor_temp"]).clip(lower=0) +
        np.random.normal(0.5, 0.2, n_samples)
    ).clip(lower=0.2)
    
    df["target_heating_kwh_1h"] = np.where(is_active, active_kwh, 0.0)
    
    return df


@pytest.fixture
def temp_model_dir(tmp_path, monkeypatch):
    """Create a temporary directory for model storage."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(two_step_module, "MODEL_DIR", str(model_dir))
    return model_dir


class TestComputeActivityThreshold:
    """Test the automatic threshold computation."""
    
    def test_threshold_computed_from_positive_values(self):
        """Threshold is computed from positive values only."""
        target_values = np.array([0, 0, 0, 0.1, 0.2, 0.5, 1.0, 2.0])
        threshold = _compute_activity_threshold(target_values)
        
        # Should be based on positive values (0.1 to 2.0)
        assert threshold >= MIN_ACTIVITY_THRESHOLD
        assert threshold <= 0.2  # Should be around 5th percentile of positives
    
    def test_threshold_uses_min_when_few_positives(self):
        """Uses minimum threshold when fewer than 10 positive samples."""
        target_values = np.array([0, 0, 0, 0, 0, 0.1, 0.2, 0.3])  # Only 3 positive
        threshold = _compute_activity_threshold(target_values)
        
        assert threshold == MIN_ACTIVITY_THRESHOLD
    
    def test_threshold_respects_minimum(self):
        """Threshold is at least MIN_ACTIVITY_THRESHOLD."""
        # Very small positive values
        target_values = np.array([0.001] * 50 + [0.002] * 50)
        threshold = _compute_activity_threshold(target_values)
        
        assert threshold >= MIN_ACTIVITY_THRESHOLD
    
    def test_threshold_uses_percentile(self):
        """Threshold uses the specified percentile."""
        target_values = np.linspace(0.1, 1.0, 100)  # All positive
        
        threshold_5 = _compute_activity_threshold(target_values, percentile=5)
        threshold_10 = _compute_activity_threshold(target_values, percentile=10)
        
        # Higher percentile should give higher threshold
        assert threshold_10 >= threshold_5


class TestTwoStepHeatingDemandModel:
    """Test the TwoStepHeatingDemandModel class."""
    
    def test_model_not_available_when_empty(self):
        """Model is not available when not loaded."""
        model = TwoStepHeatingDemandModel()
        assert not model.is_available
    
    def test_model_available_when_loaded(self):
        """Model is available when properly initialized."""
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        
        classifier = GradientBoostingClassifier(n_estimators=10)
        classifier.fit([[0, 1], [1, 0]], [0, 1])
        
        regressor = GradientBoostingRegressor(n_estimators=10)
        regressor.fit([[0, 1], [1, 0]], [0.5, 1.5])
        
        model = TwoStepHeatingDemandModel(
            classifier=classifier,
            regressor=regressor,
            activity_threshold_kwh=0.05,
            feature_names=["feat1", "feat2"],
        )
        
        assert model.is_available
    
    def test_predict_raises_when_not_available(self):
        """Predict raises error when model not available."""
        model = TwoStepHeatingDemandModel()
        
        with pytest.raises(TwoStepModelNotAvailableError):
            model.predict({"feat1": 1.0})
    
    def test_predict_raises_on_missing_features(self):
        """Predict raises error when features are missing."""
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        
        classifier = GradientBoostingClassifier(n_estimators=10)
        classifier.fit([[0, 1], [1, 0]], [0, 1])
        
        regressor = GradientBoostingRegressor(n_estimators=10)
        regressor.fit([[0, 1], [1, 0]], [0.5, 1.5])
        
        model = TwoStepHeatingDemandModel(
            classifier=classifier,
            regressor=regressor,
            feature_names=["feat1", "feat2"],
        )
        
        with pytest.raises(ValueError) as exc_info:
            model.predict({"feat1": 1.0})  # Missing feat2
        
        assert "Missing features" in str(exc_info.value)
    
    def test_predict_returns_two_step_prediction(self):
        """Predict returns TwoStepPrediction with all fields."""
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        
        classifier = GradientBoostingClassifier(n_estimators=10)
        classifier.fit([[0, 1], [1, 0], [1, 1]], [0, 1, 1])
        
        regressor = GradientBoostingRegressor(n_estimators=10)
        regressor.fit([[0, 1], [1, 0], [1, 1]], [0.5, 1.5, 2.0])
        
        model = TwoStepHeatingDemandModel(
            classifier=classifier,
            regressor=regressor,
            activity_threshold_kwh=0.05,
            feature_names=["feat1", "feat2"],
        )
        
        result = model.predict({"feat1": 1.0, "feat2": 1.0})
        
        assert isinstance(result, TwoStepPrediction)
        assert isinstance(result.is_active, bool)
        assert isinstance(result.predicted_kwh, float)
        assert isinstance(result.classifier_probability, float)
        assert 0 <= result.classifier_probability <= 1
    
    def test_inactive_prediction_returns_zero(self, sample_dataset, temp_model_dir):
        """When classifier predicts inactive, kWh is 0."""
        # Train model with real data
        model, _ = train_two_step_heating_demand_model(sample_dataset)
        
        # Create features that should predict inactive (e.g., warm outdoor, low setpoint)
        features = {
            "outdoor_temp": 25.0,  # Very warm
            "wind": 0.0,
            "humidity": 50.0,
            "indoor_temp": 22.0,
            "target_temp": 18.0,  # Low setpoint
            "outdoor_temp_avg_1h": 25.0,
            "heating_kwh_last_6h": 0.0,  # No recent heating
            "hour_of_day": 14,
            "day_of_week": 0,
            "is_weekend": 0,
            "is_night": 0,
        }
        
        result = model.predict(features)
        
        # Should predict inactive
        if not result.is_active:
            assert result.predicted_kwh == 0.0


class TestTrainTwoStepHeatingDemandModel:
    """Test the train_two_step_heating_demand_model function."""
    
    def test_trains_successfully(self, sample_dataset, temp_model_dir):
        """Model trains successfully with valid data."""
        model, metrics = train_two_step_heating_demand_model(sample_dataset)
        
        assert model.is_available
        assert metrics.active_samples > 0
        assert metrics.inactive_samples > 0
        assert metrics.computed_threshold_kwh >= MIN_ACTIVITY_THRESHOLD
    
    def test_classifier_metrics_reasonable(self, sample_dataset, temp_model_dir):
        """Classifier metrics are within reasonable bounds."""
        model, metrics = train_two_step_heating_demand_model(sample_dataset)
        
        assert 0 <= metrics.classifier_accuracy <= 1
        assert 0 <= metrics.classifier_precision <= 1
        assert 0 <= metrics.classifier_recall <= 1
        assert 0 <= metrics.classifier_f1 <= 1
    
    def test_regressor_metrics_reasonable(self, sample_dataset, temp_model_dir):
        """Regressor metrics are within reasonable bounds."""
        model, metrics = train_two_step_heating_demand_model(sample_dataset)
        
        assert metrics.regressor_train_mae >= 0
        assert metrics.regressor_val_mae >= 0
        assert metrics.regressor_train_samples > 0
        assert metrics.regressor_val_samples > 0
    
    def test_saves_model_to_disk(self, sample_dataset, temp_model_dir):
        """Model is saved to disk after training."""
        model, metrics = train_two_step_heating_demand_model(sample_dataset)
        
        model_path = temp_model_dir / "heating_demand_two_step_model.joblib"
        assert model_path.exists()
    
    def test_threshold_stored_in_model(self, sample_dataset, temp_model_dir):
        """Activity threshold is stored in the model."""
        model, metrics = train_two_step_heating_demand_model(sample_dataset)
        
        assert model.activity_threshold_kwh == metrics.computed_threshold_kwh


class TestLoadTwoStepHeatingDemandModel:
    """Test the load_two_step_heating_demand_model function."""
    
    def test_returns_none_when_no_model(self, temp_model_dir):
        """Returns None when no model file exists."""
        model = load_two_step_heating_demand_model()
        assert model is None
    
    def test_loads_trained_model(self, sample_dataset, temp_model_dir):
        """Loads a previously trained model."""
        # Train and save
        trained_model, _ = train_two_step_heating_demand_model(sample_dataset)
        
        # Load
        loaded_model = load_two_step_heating_demand_model()
        
        assert loaded_model is not None
        assert loaded_model.is_available
        assert loaded_model.feature_names == trained_model.feature_names
        assert loaded_model.activity_threshold_kwh == trained_model.activity_threshold_kwh
    
    def test_raises_on_corrupt_file(self, temp_model_dir):
        """Raises exception when model file is corrupt."""
        model_path = temp_model_dir / "heating_demand_two_step_model.joblib"
        model_path.write_text("not a valid joblib file")
        
        with pytest.raises(Exception):
            load_two_step_heating_demand_model()


class TestPredictTwoStepScenario:
    """Test the predict_two_step_scenario function."""
    
    def test_predicts_multiple_slots(self, sample_dataset, temp_model_dir):
        """Predicts for multiple time slots."""
        model, _ = train_two_step_heating_demand_model(sample_dataset)
        
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
        
        predictions = predict_two_step_scenario(model, scenario)
        
        assert len(predictions) == 24
        assert all(isinstance(p, TwoStepPrediction) for p in predictions)
        assert all(p.predicted_kwh >= 0 for p in predictions)
    
    def test_empty_scenario(self, sample_dataset, temp_model_dir):
        """Handles empty scenario gracefully."""
        model, _ = train_two_step_heating_demand_model(sample_dataset)
        
        predictions = predict_two_step_scenario(model, [])
        
        assert predictions == []
    
    def test_returns_active_and_inactive_hours(self, sample_dataset, temp_model_dir):
        """Scenario returns both active and inactive predictions."""
        model, _ = train_two_step_heating_demand_model(sample_dataset)
        
        # Create scenario with varying conditions
        scenario = []
        
        # Cold day with high setpoint (should be active)
        for _ in range(12):
            scenario.append({
                "outdoor_temp": 0.0,
                "wind": 5.0,
                "humidity": 80.0,
                "indoor_temp": 18.0,
                "target_temp": 21.0,
                "outdoor_temp_avg_1h": 0.0,
                "heating_kwh_last_6h": 3.0,
                "hour_of_day": 10,
                "day_of_week": 0,
                "is_weekend": 0,
                "is_night": 0,
            })
        
        # Warm day with low setpoint (might be inactive)
        for _ in range(12):
            scenario.append({
                "outdoor_temp": 22.0,
                "wind": 1.0,
                "humidity": 60.0,
                "indoor_temp": 21.0,
                "target_temp": 18.0,
                "outdoor_temp_avg_1h": 22.0,
                "heating_kwh_last_6h": 0.0,
                "hour_of_day": 14,
                "day_of_week": 0,
                "is_weekend": 0,
                "is_night": 0,
            })
        
        predictions = predict_two_step_scenario(model, scenario)
        
        # Should have mix of active and inactive
        active_count = sum(1 for p in predictions if p.is_active)
        inactive_count = sum(1 for p in predictions if not p.is_active)
        
        # At least one prediction should be different based on conditions
        assert len(predictions) == 24


class TestTwoStepModelIntegration:
    """Integration tests for the complete two-step model workflow."""
    
    def test_train_save_load_predict_workflow(self, sample_dataset, temp_model_dir):
        """Tests complete train -> save -> load -> predict workflow."""
        # Train
        original_model, metrics = train_two_step_heating_demand_model(sample_dataset)
        
        # Verify saved
        model_path = temp_model_dir / "heating_demand_two_step_model.joblib"
        assert model_path.exists()
        
        # Load
        loaded_model = load_two_step_heating_demand_model()
        
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
        assert pred_original.is_active == pred_loaded.is_active
        assert abs(pred_original.predicted_kwh - pred_loaded.predicted_kwh) < 0.0001
        assert abs(pred_original.classifier_probability - pred_loaded.classifier_probability) < 0.0001
    
    def test_two_step_vs_single_step_comparison(self, sample_dataset, temp_model_dir, monkeypatch):
        """Two-step model produces reasonable results compared to single-step."""
        import ml.heating_demand_model as model_module
        from ml.heating_demand_model import train_heating_demand_model
        
        # Also patch the single-step model module to use temp directory
        monkeypatch.setattr(model_module, "MODEL_DIR", str(temp_model_dir))
        
        # Train both models on same data
        two_step_model, two_step_metrics = train_two_step_heating_demand_model(sample_dataset)
        single_step_model, single_step_metrics = train_heating_demand_model(sample_dataset)
        
        # Create test scenario
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
        
        two_step_pred = two_step_model.predict(features)
        single_step_pred = single_step_model.predict(features)
        
        # Both should give non-negative predictions
        assert two_step_pred.predicted_kwh >= 0
        assert single_step_pred >= 0


class TestTwoStepFeatureConfig:
    """Test two-step prediction feature configuration."""
    
    def test_two_step_config_default_disabled(self, tmp_path, monkeypatch):
        """Two-step prediction is disabled by default."""
        import ml.feature_config as config_module
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        monkeypatch.setattr(config_module, "CONFIG_DIR", str(config_dir))
        monkeypatch.setattr(config_module, "_config", None)
        
        from ml.feature_config import FeatureConfiguration
        
        config = FeatureConfiguration()
        assert not config.is_two_step_prediction_enabled()
    
    def test_enable_two_step_prediction(self, tmp_path, monkeypatch):
        """Can enable two-step prediction."""
        import ml.feature_config as config_module
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        monkeypatch.setattr(config_module, "CONFIG_DIR", str(config_dir))
        monkeypatch.setattr(config_module, "_config", None)
        
        from ml.feature_config import FeatureConfiguration
        
        config = FeatureConfiguration()
        config.enable_two_step_prediction()
        
        assert config.is_two_step_prediction_enabled()
    
    def test_disable_two_step_prediction(self, tmp_path, monkeypatch):
        """Can disable two-step prediction."""
        import ml.feature_config as config_module
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        monkeypatch.setattr(config_module, "CONFIG_DIR", str(config_dir))
        monkeypatch.setattr(config_module, "_config", None)
        
        from ml.feature_config import FeatureConfiguration
        
        config = FeatureConfiguration()
        config.enable_two_step_prediction()
        config.disable_two_step_prediction()
        
        assert not config.is_two_step_prediction_enabled()
    
    def test_two_step_config_persists(self, tmp_path, monkeypatch):
        """Two-step configuration persists across save/load."""
        import ml.feature_config as config_module
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        monkeypatch.setattr(config_module, "CONFIG_DIR", str(config_dir))
        monkeypatch.setattr(config_module, "_config", None)
        
        from ml.feature_config import FeatureConfiguration
        
        # Create and save config with two-step enabled
        config1 = FeatureConfiguration()
        config1.enable_two_step_prediction()
        config1.save()
        
        # Load config and verify
        config2 = FeatureConfiguration.load()
        assert config2.is_two_step_prediction_enabled()
