"""
Tests for the settings optimizer module and API endpoints.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd

from app import app
from ml.optimizer import (
    run_optimization,
    apply_best_configuration,
    OptimizerProgress,
    OptimizationResult,
    _generate_experimental_feature_combinations,
    _configuration_to_name,
)


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


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
    metrics.features = ["outdoor_temp", "wind"]
    return metrics


@pytest.fixture
def mock_two_step_metrics():
    """Create mock two-step training metrics."""
    metrics = MagicMock()
    metrics.computed_threshold_kwh = 0.05
    metrics.active_samples = 250
    metrics.inactive_samples = 50
    metrics.classifier_accuracy = 0.92
    metrics.classifier_precision = 0.88
    metrics.classifier_recall = 0.95
    metrics.classifier_f1 = 0.91
    metrics.regressor_train_samples = 200
    metrics.regressor_val_samples = 50
    metrics.regressor_train_mae = 0.15
    metrics.regressor_val_mae = 0.18
    metrics.regressor_val_mape = 0.08  # Lower MAPE = better
    metrics.regressor_val_r2 = 0.85
    metrics.features = ["outdoor_temp", "wind"]
    return metrics


class TestOptimizerModule:
    """Test the optimizer module functions."""

    def test_get_experimental_feature_combinations_returns_list(self):
        """Get feature combinations returns a non-empty list."""
        combos = list(_generate_experimental_feature_combinations())
        
        assert isinstance(combos, list)
        assert len(combos) > 0
        # First combination should be all disabled (baseline)
        assert all(not v for v in combos[0].values())
    
    def test_get_experimental_feature_combinations_includes_baseline(self):
        """Get feature combinations includes baseline (all disabled)."""
        combos = list(_generate_experimental_feature_combinations())
        
        # Check baseline is first
        baseline = combos[0]
        assert all(enabled is False for enabled in baseline.values())
    
    def test_get_experimental_feature_combinations_includes_all_enabled(self):
        """Get feature combinations includes all features enabled."""
        combos = list(_generate_experimental_feature_combinations())
        
        # Last combination should be all enabled
        all_enabled = combos[-1]
        assert all(enabled is True for enabled in all_enabled.values())
    
    def test_configuration_to_name_baseline(self):
        """Configuration to name returns correct baseline name."""
        config = {"pressure": False, "outdoor_temp_avg_6h": False}
        name = _configuration_to_name(config)
        
        assert name == "Baseline (core features only)"
    
    def test_configuration_to_name_single_feature(self):
        """Configuration to name returns correct name for single feature."""
        config = {"pressure": True, "outdoor_temp_avg_6h": False}
        name = _configuration_to_name(config)
        
        assert "+pressure" in name
    
    def test_configuration_to_name_all_enabled(self):
        """Configuration to name returns correct name for all enabled."""
        config = {"pressure": True, "outdoor_temp_avg_6h": True}
        name = _configuration_to_name(config)
        
        assert name == "All features enabled"
    
    def test_run_optimization_success(self, mock_training_metrics, mock_two_step_metrics):
        """Run optimization completes successfully with mock data."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = MagicMock()
        
        mock_model = MagicMock()
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train_single(df):
            return (mock_model, mock_training_metrics)
        
        def mock_train_two_step(df):
            return (mock_model, mock_two_step_metrics)
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            progress = run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two_step,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
                configured_max_combinations=2,  # Limit combinations for fast test execution
            )
        
        assert progress.phase == "complete"
        assert len(progress.results) > 0
        assert progress.best_result is not None
    
    def test_run_optimization_finds_best_result(self, mock_training_metrics, mock_two_step_metrics):
        """Run optimization finds the configuration with lowest Val MAPE."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = MagicMock()
        
        mock_model = MagicMock()
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train_single(df):
            return (mock_model, mock_training_metrics)
        
        def mock_train_two_step(df):
            return (mock_model, mock_two_step_metrics)
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            progress = run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two_step,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
                configured_max_combinations=2,  # Limit combinations for fast test execution
            )
        
        # Two-step has lower MAPE (8%) than single-step (10%)
        assert progress.best_result.model_type == "two_step"
        assert progress.best_result.val_mape_pct == pytest.approx(8.0)
    
    def test_run_optimization_saves_and_restores_settings(self):
        """Run optimization saves original settings and restores them."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = MagicMock()
        mock_model = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.val_mape = 0.1
        mock_metrics.val_mae = 0.15
        mock_metrics.val_r2 = 0.85
        mock_metrics.train_samples = 60
        mock_metrics.val_samples = 20
        
        mock_two_step_metrics = MagicMock()
        mock_two_step_metrics.regressor_val_mape = 0.08
        mock_two_step_metrics.regressor_val_mae = 0.18
        mock_two_step_metrics.regressor_val_r2 = 0.85
        mock_two_step_metrics.regressor_train_samples = 200
        mock_two_step_metrics.regressor_val_samples = 50
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train_single(df):
            return (mock_model, mock_metrics)
        
        def mock_train_two_step(df):
            return (mock_model, mock_two_step_metrics)
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {
                "experimental_enabled": {"pressure": True}
            }
            mock_get_config.return_value = mock_config
            
            progress = run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two_step,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
                configured_max_combinations=2,  # Limit combinations for fast test execution
            )
        
        assert progress.original_settings is not None
        assert progress.original_settings["experimental_enabled"]["pressure"] is True
        # Check that restore message is in log
        assert any("Original settings restored" in msg for msg in progress.log_messages)
    
    def test_run_optimization_handles_insufficient_data(self):
        """Run optimization handles insufficient data gracefully."""
        mock_stats = MagicMock()
        
        def mock_build_dataset(min_samples):
            return (None, mock_stats)
        
        def mock_train_single(df):
            raise AssertionError("Should not be called")
        
        def mock_train_two_step(df):
            raise AssertionError("Should not be called")
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            progress = run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two_step,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
                configured_max_combinations=2,  # Limit combinations for fast test execution
            )
        
        assert progress.phase == "complete"
        # All results should be failures due to insufficient data
        assert all(not r.success for r in progress.results)
    
    def test_run_optimization_calls_progress_callback(self, mock_training_metrics, mock_two_step_metrics):
        """Run optimization calls progress callback."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = MagicMock()
        mock_model = MagicMock()
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train_single(df):
            return (mock_model, mock_training_metrics)
        
        def mock_train_two_step(df):
            return (mock_model, mock_two_step_metrics)
        
        callback_calls = []
        
        def progress_callback(progress):
            callback_calls.append(progress.phase)
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two_step,
                build_dataset_fn=mock_build_dataset,
                progress_callback=progress_callback,
                min_samples=50,
                configured_max_combinations=2,  # Limit combinations for fast test execution
            )
        
        # Callback should be called multiple times
        assert len(callback_calls) > 0
        # Last callback should be "complete"
        assert callback_calls[-1] == "complete"


class TestApplyBestConfiguration:
    """Test apply_best_configuration function."""

    def test_apply_best_configuration_success(self):
        """Apply best configuration saves settings successfully."""
        best_result = OptimizationResult(
            config_name="Test Config",
            model_type="single_step",
            experimental_features={"pressure": True, "outdoor_temp_avg_6h": False},
            val_mape_pct=10.0,
            val_mae_kwh=0.15,
            val_r2=0.85,
            train_samples=60,
            val_samples=20,
            success=True,
        )
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.save.return_value = True
            mock_get_config.return_value = mock_config
            
            result = apply_best_configuration(best_result)
        
        assert result is True
        # Updated to use enable_feature/disable_feature (handles both experimental and derived)
        mock_config.enable_feature.assert_called_once_with("pressure")
        mock_config.disable_feature.assert_called_once_with("outdoor_temp_avg_6h")
        mock_config.save.assert_called_once()
    
    def test_apply_best_configuration_enables_two_step(self):
        """Apply best configuration enables two-step for two-step model."""
        best_result = OptimizationResult(
            config_name="Test Config",
            model_type="two_step",
            experimental_features={"pressure": False},
            val_mape_pct=8.0,
            val_mae_kwh=0.18,
            val_r2=0.85,
            train_samples=200,
            val_samples=50,
            success=True,
        )
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.save.return_value = True
            mock_get_config.return_value = mock_config
            
            result = apply_best_configuration(best_result, enable_two_step=True)
        
        assert result is True
        mock_config.enable_two_step_prediction.assert_called_once()
    
    def test_apply_best_configuration_disables_two_step_for_single_step(self):
        """Apply best configuration disables two-step for single-step model."""
        best_result = OptimizationResult(
            config_name="Test Config",
            model_type="single_step",
            experimental_features={"pressure": False},
            val_mape_pct=10.0,
            val_mae_kwh=0.15,
            val_r2=0.85,
            train_samples=60,
            val_samples=20,
            success=True,
        )
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.save.return_value = True
            mock_get_config.return_value = mock_config
            
            result = apply_best_configuration(best_result, enable_two_step=True)
        
        assert result is True
        mock_config.disable_two_step_prediction.assert_called_once()


class TestOptimizerEndpoints:
    """Test the optimizer API endpoints."""

    def test_run_optimizer_success(self, client):
        """Run optimizer starts async and returns success."""
        with patch("app.threading.Thread") as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread_class.return_value = mock_thread
            
            response = client.post("/api/optimizer/run")
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["running"] is True
        assert "background" in data["message"].lower()
        
        # Verify thread was started
        mock_thread_class.assert_called_once()
        mock_thread.start.assert_called_once()
    
    def test_run_optimizer_returns_best_result(self, client):
        """Run optimizer with async - best result available via status endpoint."""
        import app as app_module
        from ml.optimizer import OptimizerProgress, OptimizationResult
        
        # Set up completed progress with best result
        result = OptimizationResult(
            config_name="Test",
            model_type="two_step",
            experimental_features={"pressure": True},
            val_mape_pct=8.0,
            val_mae_kwh=0.18,
            val_r2=0.85,
            train_samples=200,
            val_samples=50,
            success=True,
        )
        
        progress = OptimizerProgress(
            total_configurations=2,
            completed_configurations=2,
            current_configuration="",
            current_model_type="",
            phase="complete",
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        progress.results = [result]
        progress.best_result = result
        
        app_module._optimizer_progress = progress
        app_module._optimizer_running = False
        
        try:
            response = client.get("/api/optimizer/status")
            
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["running"] is False
            assert "best_result" in data["progress"]
            assert data["progress"]["best_result"]["model_type"] == "two_step"
        finally:
            app_module._optimizer_progress = None
    
    def test_get_optimizer_status_no_run(self, client):
        """Get optimizer status returns message when no run has happened."""
        # Reset global state
        import app as app_module
        app_module._optimizer_progress = None
        app_module._optimizer_running = False
        
        response = client.get("/api/optimizer/status")
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["running"] is False
        assert data["progress"] is None
    
    def test_apply_optimizer_no_results(self, client):
        """Apply optimizer returns 400 when no results."""
        # Reset global state
        import app as app_module
        app_module._optimizer_progress = None
        
        response = client.post("/api/optimizer/apply")
        
        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
        assert "No optimization results" in data["message"]
    
    def test_apply_optimizer_success(self, client):
        """Apply optimizer returns success after run."""
        import app as app_module
        from ml.optimizer import OptimizerProgress, OptimizationResult
        
        # Set up completed progress with best result
        result = OptimizationResult(
            config_name="Test",
            model_type="two_step",
            experimental_features={"pressure": True},
            val_mape_pct=8.0,
            val_mae_kwh=0.18,
            val_r2=0.85,
            train_samples=200,
            val_samples=50,
            success=True,
        )
        
        progress = OptimizerProgress(
            total_configurations=2,
            completed_configurations=2,
            current_configuration="",
            current_model_type="",
            phase="complete",
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        progress.results = [result]
        progress.best_result = result
        
        # Set the optimizer progress in the app module so apply can access it
        app_module._optimizer_progress = progress
        app_module._optimizer_running = False
        
        try:
            # Now apply
            with patch("app.apply_best_configuration") as mock_apply:
                mock_apply.return_value = True
                
                response = client.post(
                    "/api/optimizer/apply",
                    json={"enable_two_step": True},
                    content_type="application/json",
                )
            
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "applied_settings" in data
        finally:
            # Clean up
            app_module._optimizer_progress = None
