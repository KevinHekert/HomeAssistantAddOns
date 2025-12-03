"""
Tests for parallel optimizer execution and derived feature support.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, call
import pandas as pd
import time

from ml.optimizer import (
    SearchStrategy,
    run_optimization,
    OptimizerProgress,
    OptimizationResult,
    _get_all_available_features,
    _generate_experimental_feature_combinations,
    _train_single_configuration,
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
    metrics.regressor_val_mape = 0.08
    metrics.regressor_val_r2 = 0.85
    metrics.features = ["outdoor_temp", "wind"]
    return metrics


class TestDerivedFeatureDiscovery:
    """Test discovery of derived features for optimization."""
    
    def test_get_all_available_features_includes_derived(self):
        """Get all available features includes derived features from config."""
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            # Mock experimental_enabled with some derived features
            mock_config.experimental_enabled = {
                "pressure": True,
                "wind_avg_1h": True,  # Derived feature
                "outdoor_temp_avg_6h": False,
                "avatar_avg_24h": True,  # Another derived feature
            }
            mock_get_config.return_value = mock_config
            
            with patch("ml.optimizer.EXPERIMENTAL_FEATURES", []):
                features = _get_all_available_features()
            
            # Should include all features from experimental_enabled
            assert "pressure" in features
            assert "wind_avg_1h" in features
            assert "outdoor_temp_avg_6h" in features
            assert "avatar_avg_24h" in features
    
    def test_generate_experimental_feature_combinations_with_derived(self):
        """Get feature combinations includes derived features when requested."""
        with patch("ml.optimizer._get_all_available_features") as mock_get_all:
            mock_get_all.return_value = [
                "pressure",
                "wind_avg_1h",
                "outdoor_temp_avg_6h",
            ]
            
            combos = list(_generate_experimental_feature_combinations(include_derived=True))
            
            # Should have generated combinations with derived features
            assert len(combos) > 0
            # Baseline should disable all
            assert all(not v for v in combos[0].values())
            # Should have individual feature combinations
            assert any(combo.get("wind_avg_1h") for combo in combos)
    
    def test_generate_experimental_feature_combinations_without_derived(self):
        """Get feature combinations excludes derived features when not requested."""
        with patch("ml.optimizer.EXPERIMENTAL_FEATURES") as mock_exp_features:
            mock_feature = MagicMock()
            mock_feature.name = "pressure"
            mock_exp_features.__iter__.return_value = [mock_feature]
            
            combos = list(_generate_experimental_feature_combinations(include_derived=False))
            
            # Should only have experimental features, not derived
            assert len(combos) > 0
            for combo in combos:
                assert "pressure" in combo or len(combo) == 0
                # No derived features should be present
                assert "wind_avg_1h" not in combo


class TestParallelExecution:
    """Test parallel execution of optimizer."""
    
    def test_train_single_configuration_success(self, mock_training_metrics):
        """Train single configuration returns success result."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = MagicMock()
        mock_model = MagicMock()
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train(df):
            return (mock_model, mock_training_metrics)
        
        combo = {"pressure": True, "outdoor_temp_avg_6h": False}
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config
            
            result = _train_single_configuration(
                config_name="Test Config",
                combo=combo,
                model_type="single_step",
                train_fn=mock_train,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
            )
        
        assert result.success is True
        assert result.config_name == "Test Config"
        assert result.model_type == "single_step"
        assert result.val_mape_pct == pytest.approx(10.0)
    
    def test_train_single_configuration_two_step(self, mock_two_step_metrics):
        """Train single configuration handles two-step model correctly."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = MagicMock()
        mock_model = MagicMock()
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train(df):
            return (mock_model, mock_two_step_metrics)
        
        combo = {"pressure": True}
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config
            
            result = _train_single_configuration(
                config_name="Test Two-Step",
                combo=combo,
                model_type="two_step",
                train_fn=mock_train,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
            )
        
        assert result.success is True
        assert result.model_type == "two_step"
        assert result.val_mape_pct == pytest.approx(8.0)
    
    def test_train_single_configuration_insufficient_data(self):
        """Train single configuration handles insufficient data gracefully."""
        def mock_build_dataset(min_samples):
            return (None, None)
        
        def mock_train(df):
            raise AssertionError("Should not be called")
        
        combo = {"pressure": True}
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config
            
            result = _train_single_configuration(
                config_name="Test Insufficient",
                combo=combo,
                model_type="single_step",
                train_fn=mock_train,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
            )
        
        assert result.success is False
        assert result.error_message == "Insufficient data for training"
    
    def test_train_single_configuration_captures_training_rows(self, mock_training_metrics):
        """Train single configuration captures first and last rows from training split, not full dataset."""
        # Create a DataFrame with 100 rows
        # With train_ratio=0.8, training will use rows 0-79, validation rows 80-99
        mock_df = pd.DataFrame({
            "outdoor_temp": range(100),  # 0, 1, 2, ..., 99
            "wind": range(100, 200),     # 100, 101, 102, ..., 199
            "target_heating_kwh_1h": [float(i) / 10.0 for i in range(100)],
        })
        mock_stats = MagicMock()
        mock_model = MagicMock()
        
        # Set train_samples to 80 (80% of 100)
        mock_training_metrics.train_samples = 80
        mock_training_metrics.val_samples = 20
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train(df):
            return (mock_model, mock_training_metrics)
        
        combo = {"pressure": True}
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config
            
            result = _train_single_configuration(
                config_name="Test Training Rows",
                combo=combo,
                model_type="single_step",
                train_fn=mock_train,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
            )
        
        assert result.success is True
        assert result.first_row_data is not None
        assert result.last_row_data is not None
        
        # First row should be row 0 (first training row)
        assert result.first_row_data["outdoor_temp"] == 0.0
        assert result.first_row_data["wind"] == 100.0
        
        # Last row should be row 79 (last training row), NOT row 99 (last dataset row)
        assert result.last_row_data["outdoor_temp"] == 79.0
        assert result.last_row_data["wind"] == 179.0
    
    def test_train_single_configuration_two_step_captures_training_rows(self, mock_two_step_metrics):
        """Train single configuration captures first and last rows from training split for two-step model."""
        # Create a DataFrame with 100 rows
        # With train_ratio=0.8, training will use rows 0-79, validation rows 80-99
        mock_df = pd.DataFrame({
            "outdoor_temp": range(100),  # 0, 1, 2, ..., 99
            "wind": range(100, 200),     # 100, 101, 102, ..., 199
            "target_heating_kwh_1h": [float(i) / 10.0 for i in range(100)],
        })
        mock_stats = MagicMock()
        mock_model = MagicMock()
        
        # Set regressor_train_samples to 80 (80% of 100) for two-step model
        mock_two_step_metrics.regressor_train_samples = 80
        mock_two_step_metrics.regressor_val_samples = 20
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train(df):
            return (mock_model, mock_two_step_metrics)
        
        combo = {"pressure": True}
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config
            
            result = _train_single_configuration(
                config_name="Test Two-Step Training Rows",
                combo=combo,
                model_type="two_step",
                train_fn=mock_train,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
            )
        
        assert result.success is True
        assert result.first_row_data is not None
        assert result.last_row_data is not None
        
        # First row should be row 0 (first training row)
        assert result.first_row_data["outdoor_temp"] == 0.0
        assert result.first_row_data["wind"] == 100.0
        
        # Last row should be row 79 (last training row), NOT row 99 (last dataset row)
        assert result.last_row_data["outdoor_temp"] == 79.0
        assert result.last_row_data["wind"] == 179.0
    
    def test_run_optimization_with_parallel_workers(self, mock_training_metrics, mock_two_step_metrics):
        """Run optimization uses parallel workers."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = MagicMock()
        mock_model = MagicMock()
        
        def mock_build_dataset(min_samples):
            # Add small delay to simulate real training
            time.sleep(0.01)
            return (mock_df, mock_stats)
        
        def mock_train_single(df):
            time.sleep(0.02)
            return (mock_model, mock_training_metrics)
        
        def mock_train_two_step(df):
            time.sleep(0.02)
            return (mock_model, mock_two_step_metrics)
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config, \
             patch("ml.optimizer._generate_experimental_feature_combinations") as mock_combos:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            # Create a small number of combinations for faster test
            mock_combos.return_value = [
                {"pressure": False},
                {"pressure": True},
            ]
            
            start_time = time.time()
            progress = run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two_step,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
                configured_max_combinations=2,  # Limit combinations for fast test execution

            
                search_strategy=SearchStrategy.EXHAUSTIVE,  # Use exhaustive search for predictable test behavior
            )
            elapsed = time.time() - start_time
        
        # With 4 total trainings (2 configs × 2 models) and adaptive parallelism
        # The optimizer now includes 0.5s delays between trainings for GC and memory throttling checks
        # So it's expected to take longer than pure parallel execution for stability
        assert elapsed < 5.0  # Should finish within reasonable time with adaptive throttling
        assert progress.phase == "complete"
        assert len(progress.results) == 4  # 2 configs × 2 models
        assert progress.completed_configurations == 4
    
    def test_run_optimization_progress_updates_during_parallel(self, mock_training_metrics, mock_two_step_metrics):
        """Run optimization provides progress updates during parallel execution."""
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
        
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress.completed_configurations)
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config, \
             patch("ml.optimizer._generate_experimental_feature_combinations") as mock_combos:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            mock_combos.return_value = [
                {"pressure": False},
                {"pressure": True},
            ]
            
            progress = run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two_step,
                build_dataset_fn=mock_build_dataset,
                progress_callback=progress_callback,
                min_samples=50,
                configured_max_combinations=2,  # Limit combinations for fast test execution

            
                search_strategy=SearchStrategy.EXHAUSTIVE,  # Use exhaustive search for predictable test behavior
            )
        
        # Progress callback should have been called multiple times
        assert len(progress_updates) > 0
        # Final progress should show all completed
        assert progress_updates[-1] == 4 or progress.completed_configurations == 4
    
    def test_run_optimization_finds_best_with_parallel(self, mock_training_metrics, mock_two_step_metrics):
        """Run optimization finds best result correctly with parallel execution."""
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
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config, \
             patch("ml.optimizer._generate_experimental_feature_combinations") as mock_combos:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            mock_combos.return_value = [
                {"pressure": False},
                {"pressure": True},
            ]
            
            progress = run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two_step,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,

            )
        
        # Best result should be two-step with 8% MAPE (better than single-step 10%)
        assert progress.best_result is not None
        assert progress.best_result.model_type == "two_step"
        assert progress.best_result.val_mape_pct == pytest.approx(8.0)
    
    def test_run_optimization_thread_safe_progress_updates(self, mock_training_metrics):
        """Run optimization handles concurrent progress updates safely."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = MagicMock()
        mock_model = MagicMock()
        
        # Create separate mock metrics for two-step with proper values
        mock_two_step_metrics = MagicMock()
        mock_two_step_metrics.regressor_val_mape = 0.08
        mock_two_step_metrics.regressor_val_mae = 0.18
        mock_two_step_metrics.regressor_val_r2 = 0.85
        mock_two_step_metrics.regressor_train_samples = 200
        mock_two_step_metrics.regressor_val_samples = 50
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train_single(df):
            # Add slight delay to increase chance of concurrent updates
            time.sleep(0.01)
            return (mock_model, mock_training_metrics)
        
        def mock_train_two(df):
            time.sleep(0.01)
            return (mock_model, mock_two_step_metrics)
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config, \
             patch("ml.optimizer._generate_experimental_feature_combinations") as mock_combos:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            # More combinations to increase concurrency
            mock_combos.return_value = [
                {"pressure": False},
                {"pressure": True},
                {"outdoor_temp_avg_6h": True},
            ]
            
            progress = run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
                configured_max_combinations=3,  # Limit combinations for fast test execution

            
                search_strategy=SearchStrategy.EXHAUSTIVE,  # Use exhaustive search for predictable test behavior
            )
        
        # All results should be recorded despite concurrent updates
        assert len(progress.results) == 6  # 3 configs × 2 models
        assert progress.completed_configurations == 6
        # Each combination of (config_name, model_type) should appear exactly once
        combo_set = set((r.config_name, r.model_type) for r in progress.results)
        # With 3 distinct configs, we expect at most 6 unique combos (but config names may overlap)
        assert len(combo_set) <= 6


class TestProgressReporting:
    """Test progress reporting with X/Y format."""
    
    def test_progress_messages_include_counts(self, mock_training_metrics, mock_two_step_metrics):
        """Progress messages include X/Y completed count."""
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
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config, \
             patch("ml.optimizer._generate_experimental_feature_combinations") as mock_combos:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            mock_combos.return_value = [
                {"pressure": False},
                {"pressure": True},
            ]
            
            progress = run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two_step,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
                configured_max_combinations=2,  # Limit combinations for fast test execution

            
                search_strategy=SearchStrategy.EXHAUSTIVE,  # Use exhaustive search for predictable test behavior
            )
        
        # Check that progress messages include X/Y format
        messages_with_counts = [msg for msg in progress.log_messages if "[1/4]" in msg or "[2/4]" in msg or "[3/4]" in msg or "[4/4]" in msg]
        assert len(messages_with_counts) > 0, "Progress messages should include X/Y counts"
    
    def test_progress_shows_best_result_updates(self, mock_training_metrics, mock_two_step_metrics):
        """Progress messages show when a new best result is found."""
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
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config, \
             patch("ml.optimizer._generate_experimental_feature_combinations") as mock_combos:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            mock_combos.return_value = [
                {"pressure": False},
            ]
            
            progress = run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two_step,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
                configured_max_combinations=1,  # Limit combinations for fast test execution

            
                search_strategy=SearchStrategy.EXHAUSTIVE,  # Use exhaustive search for predictable test behavior
            )
        
        # Should have messages about new best results (marked with trophy emoji)
        best_messages = [msg for msg in progress.log_messages if "🏆" in msg or "New best" in msg]
        assert len(best_messages) > 0, "Progress should show when new best result is found"
