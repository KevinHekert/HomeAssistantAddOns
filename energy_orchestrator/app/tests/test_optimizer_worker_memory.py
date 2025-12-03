"""
Tests for optimizer worker memory reporting.
"""

import pytest
from unittest.mock import patch, MagicMock, call
import pandas as pd

from ml.optimizer import (
    SearchStrategy,
    _train_single_configuration,
    run_optimization,
    OptimizationResult,
)


class TestWorkerMemoryReporting:
    """Test per-worker memory reporting functionality."""
    
    def test_train_single_configuration_reports_memory_before_training(self):
        """Training function reports worker memory before training."""
        with patch('ml.optimizer.get_feature_config'), \
             patch('ml.optimizer._log_memory_usage') as mock_log_memory, \
             patch('ml.optimizer.gc.collect'):
            
            # Mock dataset building and training
            mock_build_dataset = MagicMock()
            mock_df = pd.DataFrame({"outdoor_temp": [10, 15, 20]})
            mock_build_dataset.return_value = (mock_df, None)
            
            mock_train = MagicMock()
            mock_metrics = MagicMock()
            mock_metrics.train_samples = 60
            mock_metrics.val_samples = 20
            mock_metrics.val_mae = 0.15
            mock_metrics.val_mape = 0.10
            mock_metrics.val_r2 = 0.85
            mock_train.return_value = (MagicMock(), mock_metrics)
            
            # Set up memory usage return value
            mock_log_memory.return_value = {"rss_mb": 100.0}
            
            # Call training function
            result = _train_single_configuration(
                config_name="Test",
                combo={"feature1": True},
                model_type="single_step",
                train_fn=mock_train,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
            )
            
            # Verify memory was logged before training
            assert mock_log_memory.call_count >= 2  # At least before and after
            calls = mock_log_memory.call_args_list
            assert any("BEFORE" in str(call) for call in calls)
    
    def test_train_single_configuration_reports_memory_after_training(self):
        """Training function reports worker memory after training."""
        with patch('ml.optimizer.get_feature_config'), \
             patch('ml.optimizer._log_memory_usage') as mock_log_memory, \
             patch('ml.optimizer.gc.collect'):
            
            # Mock dataset building and training
            mock_build_dataset = MagicMock()
            mock_df = pd.DataFrame({"outdoor_temp": [10, 15, 20]})
            mock_build_dataset.return_value = (mock_df, None)
            
            mock_train = MagicMock()
            mock_metrics = MagicMock()
            mock_metrics.train_samples = 60
            mock_metrics.val_samples = 20
            mock_metrics.val_mae = 0.15
            mock_metrics.val_mape = 0.10
            mock_metrics.val_r2 = 0.85
            mock_train.return_value = (MagicMock(), mock_metrics)
            
            # Set up memory usage return value
            mock_log_memory.return_value = {"rss_mb": 150.0}
            
            # Call training function
            result = _train_single_configuration(
                config_name="Test",
                combo={"feature1": True},
                model_type="single_step",
                train_fn=mock_train,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
            )
            
            # Verify memory was logged after training
            calls = mock_log_memory.call_args_list
            assert any("AFTER" in str(call) for call in calls)
    
    def test_train_single_configuration_reports_memory_after_cleanup(self):
        """Training function reports worker memory after cleanup."""
        with patch('ml.optimizer.get_feature_config'), \
             patch('ml.optimizer._log_memory_usage') as mock_log_memory, \
             patch('ml.optimizer.gc.collect'):
            
            # Mock dataset building and training
            mock_build_dataset = MagicMock()
            mock_df = pd.DataFrame({"outdoor_temp": [10, 15, 20]})
            mock_build_dataset.return_value = (mock_df, None)
            
            mock_train = MagicMock()
            mock_metrics = MagicMock()
            mock_metrics.train_samples = 60
            mock_metrics.val_samples = 20
            mock_metrics.val_mae = 0.15
            mock_metrics.val_mape = 0.10
            mock_metrics.val_r2 = 0.85
            mock_train.return_value = (MagicMock(), mock_metrics)
            
            # Set up memory usage return values
            mock_log_memory.side_effect = [
                {"rss_mb": 100.0},  # Before
                {"rss_mb": 150.0},  # After
                {"rss_mb": 110.0},  # After cleanup
            ]
            
            # Call training function
            result = _train_single_configuration(
                config_name="Test",
                combo={"feature1": True},
                model_type="single_step",
                train_fn=mock_train,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
            )
            
            # Verify memory was logged 3 times: before, after, after cleanup
            assert mock_log_memory.call_count == 3
            calls = mock_log_memory.call_args_list
            assert "BEFORE" in str(calls[0])
            assert "AFTER" in str(calls[1])
            assert "cleanup" in str(calls[2])
    
    def test_train_single_configuration_calls_gc_collect(self):
        """Training function calls garbage collection after training."""
        with patch('ml.optimizer.get_feature_config'), \
             patch('ml.optimizer._log_memory_usage') as mock_log_memory, \
             patch('ml.optimizer.gc.collect') as mock_gc:
            
            # Mock dataset building and training
            mock_build_dataset = MagicMock()
            mock_df = pd.DataFrame({"outdoor_temp": [10, 15, 20]})
            mock_build_dataset.return_value = (mock_df, None)
            
            mock_train = MagicMock()
            mock_metrics = MagicMock()
            mock_metrics.train_samples = 60
            mock_metrics.val_samples = 20
            mock_metrics.val_mae = 0.15
            mock_metrics.val_mape = 0.10
            mock_metrics.val_r2 = 0.85
            mock_train.return_value = (MagicMock(), mock_metrics)
            
            mock_log_memory.return_value = {"rss_mb": 100.0}
            
            # Call training function
            result = _train_single_configuration(
                config_name="Test",
                combo={"feature1": True},
                model_type="single_step",
                train_fn=mock_train,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
            )
            
            # Verify garbage collection was called
            mock_gc.assert_called()


class TestConfiguredMaxWorkers:
    """Test configured max_workers parameter functionality."""
    
    def test_run_optimization_uses_configured_max_workers(self):
        """Optimizer uses configured max_workers when provided."""
        with patch('ml.optimizer.get_feature_config') as mock_get_config, \
             patch('ml.optimizer._generate_experimental_feature_combinations') as mock_combos, \
             patch('ml.optimizer.ThreadPoolExecutor') as mock_executor, \
             patch('ml.optimizer._log_memory_usage'), \
             patch('ml.optimizer._Logger') as mock_logger:
            
            # Mock feature config
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            # Mock minimal feature combinations
            mock_combos.return_value = [{"feature1": False}]
            
            # Mock executor to avoid actual execution
            mock_executor_instance = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            mock_executor_instance.submit.return_value = MagicMock()
            
            # Mock training functions
            mock_train = MagicMock()
            mock_build = MagicMock()
            
            # Run optimization with configured max_workers
            try:
                progress = run_optimization(
                    train_single_step_fn=mock_train,
                    train_two_step_fn=mock_train,
                    build_dataset_fn=mock_build,
                    progress_callback=None,
                    min_samples=50,
                    include_derived_features=False,
                    max_memory_mb=None,
                    configured_max_workers=5,
                    configured_max_combinations=2,  # Limit combinations for fast test execution
                
                search_strategy=SearchStrategy.EXHAUSTIVE,  # Use exhaustive search for predictable test behavior
            )
            except Exception:
                # Expected to fail due to mocking, but we just want to check the logger call
                pass
            
            # Verify the logger was informed about configured workers
            log_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("configured max_workers: 5" in call.lower() for call in log_calls)
    
    def test_run_optimization_auto_calculates_when_max_workers_is_none(self):
        """Optimizer auto-calculates workers when configured_max_workers is None."""
        with patch('ml.optimizer.get_feature_config') as mock_get_config, \
             patch('ml.optimizer._generate_experimental_feature_combinations') as mock_combos, \
             patch('ml.optimizer._calculate_optimal_workers') as mock_calc, \
             patch('ml.optimizer.ThreadPoolExecutor') as mock_executor, \
             patch('ml.optimizer._log_memory_usage'):
            
            # Mock feature config
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            # Mock minimal feature combinations
            mock_combos.return_value = [{"feature1": False}]
            mock_calc.return_value = 3
            
            # Mock executor to avoid actual execution
            mock_executor_instance = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            mock_executor_instance.submit.return_value = MagicMock()
            
            # Mock training functions
            mock_train = MagicMock()
            mock_build = MagicMock()
            
            # Run optimization with None configured_max_workers
            try:
                progress = run_optimization(
                    train_single_step_fn=mock_train,
                    train_two_step_fn=mock_train,
                    build_dataset_fn=mock_build,
                    progress_callback=None,
                    min_samples=50,
                    include_derived_features=False,
                    max_memory_mb=None,
                    configured_max_workers=None,
                    configured_max_combinations=2,  # Limit combinations for fast test execution
                
                search_strategy=SearchStrategy.EXHAUSTIVE,  # Use exhaustive search for predictable test behavior
            )
            except Exception:
                # Expected to fail due to mocking, but we just want to check calc was called
                pass
            
            # Verify auto-calculation was called
            mock_calc.assert_called_once()
    
    def test_run_optimization_auto_calculates_when_max_workers_is_zero(self):
        """Optimizer auto-calculates workers when configured_max_workers is 0."""
        with patch('ml.optimizer.get_feature_config') as mock_get_config, \
             patch('ml.optimizer._generate_experimental_feature_combinations') as mock_combos, \
             patch('ml.optimizer._calculate_optimal_workers') as mock_calc, \
             patch('ml.optimizer.ThreadPoolExecutor') as mock_executor, \
             patch('ml.optimizer._log_memory_usage'):
            
            # Mock feature config
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_get_config.return_value = mock_config
            
            # Mock minimal feature combinations
            mock_combos.return_value = [{"feature1": False}]
            mock_calc.return_value = 3
            
            # Mock executor to avoid actual execution
            mock_executor_instance = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            mock_executor_instance.submit.return_value = MagicMock()
            
            # Mock training functions
            mock_train = MagicMock()
            mock_build = MagicMock()
            
            # Run optimization with 0 configured_max_workers
            try:
                progress = run_optimization(
                    train_single_step_fn=mock_train,
                    train_two_step_fn=mock_train,
                    build_dataset_fn=mock_build,
                    progress_callback=None,
                    min_samples=50,
                    include_derived_features=False,
                    max_memory_mb=None,
                    configured_max_workers=0,
                    configured_max_combinations=2,  # Limit combinations for fast test execution
                
                search_strategy=SearchStrategy.EXHAUSTIVE,  # Use exhaustive search for predictable test behavior
            )
            except Exception:
                # Expected to fail due to mocking, but we just want to check calc was called
                pass
            
            # Verify auto-calculation was called
            mock_calc.assert_called_once()
