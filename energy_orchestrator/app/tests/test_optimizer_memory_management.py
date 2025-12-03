"""
Tests for optimizer memory management features.

These tests verify that the optimizer properly cleans up memory during
long-running optimization runs to prevent OOM kills.
"""

import pytest
from unittest.mock import patch, MagicMock, call
import gc

import pandas as pd
import numpy as np

from ml.optimizer import (
    _train_single_configuration,
    OptimizationResult,
)


@pytest.fixture
def mock_training_metrics():
    """Create mock single-step training metrics."""
    metrics = MagicMock()
    metrics.train_samples = 60
    metrics.val_samples = 20
    metrics.train_mae = 0.1
    metrics.val_mae = 0.15
    metrics.val_mape = 0.10
    metrics.val_r2 = 0.85
    return metrics


@pytest.fixture
def mock_dataset():
    """Create a mock dataset (DataFrame) for testing."""
    # Create a realistic DataFrame similar to what the model would use
    n_rows = 1000
    df = pd.DataFrame({
        'outdoor_temp': np.random.randn(n_rows),
        'wind': np.random.randn(n_rows),
        'humidity': np.random.randn(n_rows),
        'target_heating_kwh_1h': np.random.rand(n_rows),
    })
    return df


@pytest.fixture
def mock_model():
    """Create a mock model object."""
    model = MagicMock()
    model.predict = MagicMock(return_value=np.array([0.5, 0.6, 0.7]))
    return model


class TestOptimizerMemoryManagement:
    """Test memory management in the optimizer."""

    def test_train_single_configuration_deletes_dataframe(
        self, mock_dataset, mock_model, mock_training_metrics
    ):
        """Verify that DataFrame is deleted after training to free memory."""
        config_name = "Test Config"
        combo = {"feature1": True, "feature2": False}
        
        # Mock the training function to return model and metrics
        def mock_train_fn(df):
            # Verify df is the expected DataFrame
            assert isinstance(df, pd.DataFrame)
            return mock_model, mock_training_metrics
        
        # Mock the dataset building function
        def mock_build_dataset_fn(min_samples):
            return mock_dataset, MagicMock()
        
        # Track if del was called on the correct objects
        # We can't directly test del, but we can verify gc.collect() is called
        with patch('ml.optimizer.gc.collect') as mock_gc_collect:
            result = _train_single_configuration(
                config_name=config_name,
                combo=combo,
                model_type="single_step",
                train_fn=mock_train_fn,
                build_dataset_fn=mock_build_dataset_fn,
                min_samples=50,
            )
            
            # Verify that gc.collect() was called to free memory
            assert mock_gc_collect.called
            assert mock_gc_collect.call_count >= 1
    
    def test_train_single_configuration_calls_gc_collect(
        self, mock_dataset, mock_model, mock_training_metrics
    ):
        """Verify that gc.collect() is called after training."""
        config_name = "Test Config"
        combo = {"feature1": True}
        
        def mock_train_fn(df):
            return mock_model, mock_training_metrics
        
        def mock_build_dataset_fn(min_samples):
            return mock_dataset, MagicMock()
        
        with patch('ml.optimizer.gc.collect') as mock_gc_collect:
            result = _train_single_configuration(
                config_name=config_name,
                combo=combo,
                model_type="single_step",
                train_fn=mock_train_fn,
                build_dataset_fn=mock_build_dataset_fn,
                min_samples=50,
            )
            
            # gc.collect() should be called exactly once per training
            assert mock_gc_collect.call_count == 1
    
    def test_train_single_configuration_gc_called_for_two_step(
        self, mock_dataset, mock_model
    ):
        """Verify gc.collect() is called for two-step model training."""
        config_name = "Two-Step Config"
        combo = {"feature1": True}
        
        # Mock two-step metrics
        metrics = MagicMock()
        metrics.regressor_train_samples = 200
        metrics.regressor_val_samples = 50
        metrics.regressor_train_mae = 0.15
        metrics.regressor_val_mae = 0.18
        metrics.regressor_val_mape = 0.08
        metrics.regressor_val_r2 = 0.85
        
        def mock_train_fn(df):
            return mock_model, metrics
        
        def mock_build_dataset_fn(min_samples):
            return mock_dataset, MagicMock()
        
        with patch('ml.optimizer.gc.collect') as mock_gc_collect:
            result = _train_single_configuration(
                config_name=config_name,
                combo=combo,
                model_type="two_step",
                train_fn=mock_train_fn,
                build_dataset_fn=mock_build_dataset_fn,
                min_samples=50,
            )
            
            # gc.collect() should be called for two-step models too
            assert mock_gc_collect.call_count == 1
            assert result.success is True
    
    def test_train_single_configuration_gc_not_called_on_insufficient_data(self):
        """Verify gc.collect() is not called when dataset building fails."""
        config_name = "Failed Config"
        combo = {"feature1": True}
        
        def mock_train_fn(df):
            # Should not be called
            raise AssertionError("Train function should not be called")
        
        def mock_build_dataset_fn(min_samples):
            # Return None to simulate insufficient data
            return None, MagicMock()
        
        with patch('ml.optimizer.gc.collect') as mock_gc_collect:
            result = _train_single_configuration(
                config_name=config_name,
                combo=combo,
                model_type="single_step",
                train_fn=mock_train_fn,
                build_dataset_fn=mock_build_dataset_fn,
                min_samples=50,
            )
            
            # gc.collect() should not be called if training didn't happen
            assert mock_gc_collect.call_count == 0
            assert result.success is False
            assert result.error_message == "Insufficient data for training"
    
    def test_train_single_configuration_gc_not_called_on_exception(self):
        """Verify gc.collect() is not called when training raises exception."""
        config_name = "Exception Config"
        combo = {"feature1": True}
        
        def mock_train_fn(df):
            raise ValueError("Training failed")
        
        def mock_build_dataset_fn(min_samples):
            return pd.DataFrame({'col': [1, 2, 3]}), MagicMock()
        
        with patch('ml.optimizer.gc.collect') as mock_gc_collect:
            result = _train_single_configuration(
                config_name=config_name,
                combo=combo,
                model_type="single_step",
                train_fn=mock_train_fn,
                build_dataset_fn=mock_build_dataset_fn,
                min_samples=50,
            )
            
            # gc.collect() should not be called when exception occurs
            assert mock_gc_collect.call_count == 0
            assert result.success is False
            assert "Training failed" in result.error_message
