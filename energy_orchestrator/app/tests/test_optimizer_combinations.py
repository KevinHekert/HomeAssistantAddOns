"""
Tests for optimizer combination generation.

This test module verifies that:
1. The optimizer generates ALL possible combinations (2^N)
2. Specific combinations mentioned in the issue are included
3. Each parallel run uses different feature settings (thread safety)
4. Top N results can be retrieved
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from collections import Counter
from math import comb

from ml.optimizer import (
    _get_experimental_feature_combinations,
    run_optimization,
    _train_single_configuration,
    OptimizerProgress,
    OptimizationResult,
)
from ml.feature_config import EXPERIMENTAL_FEATURES


class TestCombinationGeneration:
    """Test the combination generation logic."""

    def test_generates_all_combinations(self):
        """Verify that ALL combinations (2^N) are generated."""
        combos = _get_experimental_feature_combinations(include_derived=False)
        
        n_features = len(EXPERIMENTAL_FEATURES)
        expected_count = 2 ** n_features
        
        assert len(combos) == expected_count, f"Expected {expected_count} combinations (2^{n_features}), got {len(combos)}"
    
    def test_combination_distribution(self):
        """Verify the distribution of combinations by size."""
        combos = _get_experimental_feature_combinations(include_derived=False)
        
        # Count combinations by number of enabled features
        enabled_counts = Counter()
        for combo in combos:
            count = sum(1 for v in combo.values() if v)
            enabled_counts[count] += 1
        
        n_features = len(EXPERIMENTAL_FEATURES)
        
        # Verify combinatorial counts: C(n, k) = n! / (k! * (n-k)!)
        for size in range(n_features + 1):
            expected = comb(n_features, size)
            actual = enabled_counts[size]
            assert actual == expected, f"Size {size}: expected {expected}, got {actual}"
    
    def test_specific_pairwise_combinations_exist(self):
        """Verify specific 2-feature combinations mentioned in the issue."""
        combos = _get_experimental_feature_combinations(include_derived=False)
        
        # These specific combinations were mentioned in the issue
        required_pairs = [
            {'target_temp_avg_24h', 'heating_degree_hours_7d'},
            {'target_temp_avg_24h', 'heating_degree_hours_24h'},
            {'day_of_week', 'is_weekend'},
            {'pressure', 'outdoor_temp_avg_6h'},
        ]
        
        for required_pair in required_pairs:
            found = False
            for combo in combos:
                enabled = {k for k, v in combo.items() if v}
                if enabled == required_pair:
                    found = True
                    break
            assert found, f"Required pair {required_pair} not found in combinations"
    
    def test_individual_features_exist(self):
        """Verify individual features are tested separately."""
        combos = _get_experimental_feature_combinations(include_derived=False)
        
        # These individual features were mentioned in the issue
        individual_features = ['day_of_week', 'is_weekend', 'is_night']
        
        for feature in individual_features:
            found = False
            for combo in combos:
                enabled = {k for k, v in combo.items() if v}
                if enabled == {feature}:
                    found = True
                    break
            assert found, f"Individual feature {feature} not found in combinations"
    
    def test_baseline_and_all_enabled_exist(self):
        """Verify baseline and all-enabled combinations exist."""
        combos = _get_experimental_feature_combinations(include_derived=False)
        
        # Baseline (all disabled)
        found_baseline = any(
            all(not v for v in combo.values())
            for combo in combos
        )
        assert found_baseline, "Baseline combination not found"
        
        # All enabled
        found_all_enabled = any(
            all(v for v in combo.values())
            for combo in combos
        )
        assert found_all_enabled, "All-enabled combination not found"
    
    def test_no_duplicate_combinations(self):
        """Verify no duplicate combinations are generated."""
        combos = _get_experimental_feature_combinations(include_derived=False)
        
        # Convert each combo to a frozenset of enabled features for comparison
        combo_sets = set(
            frozenset(k for k, v in combo.items() if v)
            for combo in combos
        )
        
        # Check for duplicates
        assert len(combo_sets) == len(combos), "Duplicate combinations found"
    
    def test_all_size_3_combinations_exist(self):
        """Verify all 3-feature combinations exist (not just logical groups)."""
        combos = _get_experimental_feature_combinations(include_derived=False)
        
        feature_names = [f.name for f in EXPERIMENTAL_FEATURES]
        
        # Generate all possible 3-feature combinations
        from itertools import combinations as iter_combinations
        all_3_combos = list(iter_combinations(feature_names, 3))
        
        # Check that all 3-feature combinations are in the generated combos
        for three_features in all_3_combos:
            expected_set = set(three_features)
            found = False
            for combo in combos:
                enabled = {k for k, v in combo.items() if v}
                if enabled == expected_set:
                    found = True
                    break
            assert found, f"3-feature combination {expected_set} not found"
    
    def test_all_size_4_combinations_exist(self):
        """Verify all 4-feature combinations exist."""
        combos = _get_experimental_feature_combinations(include_derived=False)
        
        feature_names = [f.name for f in EXPERIMENTAL_FEATURES]
        
        # Generate all possible 4-feature combinations
        from itertools import combinations as iter_combinations
        all_4_combos = list(iter_combinations(feature_names, 4))
        
        # Check that all 4-feature combinations are in the generated combos
        for four_features in all_4_combos:
            expected_set = set(four_features)
            found = False
            for combo in combos:
                enabled = {k for k, v in combo.items() if v}
                if enabled == expected_set:
                    found = True
                    break
            assert found, f"4-feature combination {expected_set} not found"


class TestTopResults:
    """Test the get_top_results functionality."""
    
    def test_get_top_results_returns_correct_count(self):
        """Verify get_top_results returns the requested number of results."""
        # Create progress with some results
        progress = OptimizerProgress(
            total_configurations=10,
            completed_configurations=10,
            current_configuration="",
            current_model_type="",
            phase="complete",
        )
        
        # Add 30 results with varying MAPE values
        for i in range(30):
            result = OptimizationResult(
                config_name=f"Config {i}",
                model_type="single_step",
                experimental_features={},
                val_mape_pct=float(i + 10),  # MAPE from 10% to 39%
                val_mae_kwh=0.15,
                val_r2=0.85,
                train_samples=60,
                val_samples=20,
                success=True,
            )
            progress.results.append(result)
        
        # Get top 20
        top_20 = progress.get_top_results(20)
        
        assert len(top_20) == 20, f"Expected 20 results, got {len(top_20)}"
    
    def test_get_top_results_sorted_by_mape(self):
        """Verify results are sorted by MAPE (ascending)."""
        progress = OptimizerProgress(
            total_configurations=10,
            completed_configurations=10,
            current_configuration="",
            current_model_type="",
            phase="complete",
        )
        
        # Add results with random MAPE values
        mape_values = [25.5, 10.2, 30.1, 15.7, 20.3, 12.8, 18.5, 22.1, 14.3, 16.9]
        for i, mape in enumerate(mape_values):
            result = OptimizationResult(
                config_name=f"Config {i}",
                model_type="single_step",
                experimental_features={},
                val_mape_pct=mape,
                val_mae_kwh=0.15,
                val_r2=0.85,
                train_samples=60,
                val_samples=20,
                success=True,
            )
            progress.results.append(result)
        
        # Get top 5
        top_5 = progress.get_top_results(5)
        
        # Verify they are sorted (ascending MAPE)
        mapes = [r.val_mape_pct for r in top_5]
        assert mapes == sorted(mapes), f"Results not sorted: {mapes}"
        
        # Verify we got the lowest 5
        expected_top_5 = sorted(mape_values)[:5]
        assert mapes == expected_top_5, f"Expected {expected_top_5}, got {mapes}"
    
    def test_get_top_results_filters_failures(self):
        """Verify failed results are excluded."""
        progress = OptimizerProgress(
            total_configurations=10,
            completed_configurations=10,
            current_configuration="",
            current_model_type="",
            phase="complete",
        )
        
        # Add 5 successful and 5 failed results
        for i in range(5):
            progress.results.append(OptimizationResult(
                config_name=f"Success {i}",
                model_type="single_step",
                experimental_features={},
                val_mape_pct=float(i + 10),
                val_mae_kwh=0.15,
                val_r2=0.85,
                train_samples=60,
                val_samples=20,
                success=True,
            ))
            progress.results.append(OptimizationResult(
                config_name=f"Failed {i}",
                model_type="single_step",
                experimental_features={},
                val_mape_pct=None,
                val_mae_kwh=None,
                val_r2=None,
                train_samples=0,
                val_samples=0,
                success=False,
                error_message="Insufficient data",
            ))
        
        # Get top results
        top_results = progress.get_top_results(10)
        
        # Should only have the 5 successful results
        assert len(top_results) == 5, f"Expected 5 successful results, got {len(top_results)}"
        assert all(r.success for r in top_results), "Failed results were included"


class TestLogTailLimit:
    """Test that log messages are limited to the last N entries."""
    
    def test_log_tail_keeps_only_last_10_messages(self):
        """Verify that only the last 10 log messages are kept."""
        progress = OptimizerProgress(
            total_configurations=20,
            completed_configurations=0,
            current_configuration="",
            current_model_type="",
            phase="training",
            max_log_messages=10,
        )
        
        # Add 20 log messages
        for i in range(20):
            progress.add_log_message(f"Log message {i}")
        
        # Should only have the last 10
        assert len(progress.log_messages) == 10, f"Expected 10 messages, got {len(progress.log_messages)}"
        
        # Verify they are the LAST 10 (messages 10-19)
        expected = [f"Log message {i}" for i in range(10, 20)]
        assert progress.log_messages == expected, f"Expected {expected}, got {progress.log_messages}"
    
    def test_log_tail_with_fewer_than_max_messages(self):
        """Verify that with fewer than max messages, all are kept."""
        progress = OptimizerProgress(
            total_configurations=5,
            completed_configurations=0,
            current_configuration="",
            current_model_type="",
            phase="training",
            max_log_messages=10,
        )
        
        # Add 5 log messages (less than max)
        for i in range(5):
            progress.add_log_message(f"Log message {i}")
        
        # Should have all 5
        assert len(progress.log_messages) == 5, f"Expected 5 messages, got {len(progress.log_messages)}"
        expected = [f"Log message {i}" for i in range(5)]
        assert progress.log_messages == expected
    
    def test_log_tail_with_custom_limit(self):
        """Verify that custom log tail limit works."""
        progress = OptimizerProgress(
            total_configurations=10,
            completed_configurations=0,
            current_configuration="",
            current_model_type="",
            phase="training",
            max_log_messages=3,  # Custom limit of 3
        )
        
        # Add 10 log messages
        for i in range(10):
            progress.add_log_message(f"Message {i}")
        
        # Should only have the last 3
        assert len(progress.log_messages) == 3, f"Expected 3 messages, got {len(progress.log_messages)}"
        expected = ["Message 7", "Message 8", "Message 9"]
        assert progress.log_messages == expected


class TestThreadSafety:
    """Test that parallel training uses correct feature settings."""
    
    def test_each_training_uses_correct_features(self):
        """Verify that each parallel training run uses its own feature configuration."""
        # Track which features were configured for each training
        configured_features = []
        
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0, 12.0],
            "target_heating_kwh_1h": [1.0, 1.5, 1.2],
        })
        mock_stats = MagicMock()
        mock_model = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.val_mape = 0.1
        mock_metrics.val_mae = 0.15
        mock_metrics.val_r2 = 0.85
        mock_metrics.train_samples = 60
        mock_metrics.val_samples = 20
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train_single(df):
            return (mock_model, mock_metrics)
        
        # Test with 3 specific combinations
        test_combos = [
            {"pressure": True, "outdoor_temp_avg_6h": False},
            {"pressure": False, "outdoor_temp_avg_6h": True},
            {"pressure": True, "outdoor_temp_avg_6h": True},
        ]
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.experimental_enabled = {}
            
            # Track calls to enable_feature and disable_feature
            enable_calls = []
            disable_calls = []
            
            def track_enable(feature_name):
                enable_calls.append(feature_name)
                mock_config.experimental_enabled[feature_name] = True
            
            def track_disable(feature_name):
                disable_calls.append(feature_name)
                mock_config.experimental_enabled[feature_name] = False
            
            mock_config.enable_feature.side_effect = track_enable
            mock_config.disable_feature.side_effect = track_disable
            mock_get_config.return_value = mock_config
            
            for i, combo in enumerate(test_combos):
                # Reset tracking for each configuration
                enable_calls.clear()
                disable_calls.clear()
                
                result = _train_single_configuration(
                    config_name=f"Test {i}",
                    combo=combo,
                    model_type="single_step",
                    train_fn=mock_train_single,
                    build_dataset_fn=mock_build_dataset,
                    min_samples=50,
                )
                assert result.success, f"Training {i} failed"
                
                # Store what was configured
                configured = {
                    'enabled': enable_calls.copy(),
                    'disabled': disable_calls.copy(),
                    'combo': combo.copy()
                }
                configured_features.append(configured)
        
        # Verify that each training configured different features
        assert len(configured_features) == 3, "Should have captured 3 configurations"
        
        # Verify each combination was applied correctly
        # Training 0: pressure=True, outdoor_temp_avg_6h=False
        assert "pressure" in configured_features[0]['enabled']
        assert "outdoor_temp_avg_6h" in configured_features[0]['disabled']
        
        # Training 1: pressure=False, outdoor_temp_avg_6h=True
        assert "pressure" in configured_features[1]['disabled']
        assert "outdoor_temp_avg_6h" in configured_features[1]['enabled']
        
        # Training 2: pressure=True, outdoor_temp_avg_6h=True
        assert "pressure" in configured_features[2]['enabled']
        assert "outdoor_temp_avg_6h" in configured_features[2]['enabled']
    
    def test_parallel_training_with_different_configs(self):
        """Test that parallel execution maintains separate feature configs."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0, 12.0],
            "target_heating_kwh_1h": [1.0, 1.5, 1.2],
        })
        mock_stats = MagicMock()
        mock_model = MagicMock()
        
        # Create metrics that vary based on features
        def create_metrics(val_mape):
            metrics = MagicMock()
            metrics.val_mape = val_mape
            metrics.val_mae = 0.15
            metrics.val_r2 = 0.85
            metrics.train_samples = 60
            metrics.val_samples = 20
            metrics.regressor_val_mape = val_mape
            metrics.regressor_val_mae = 0.15
            metrics.regressor_val_r2 = 0.85
            metrics.regressor_train_samples = 60
            metrics.regressor_val_samples = 20
            return metrics
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train_single(df):
            return (mock_model, create_metrics(0.10))
        
        def mock_train_two_step(df):
            return (mock_model, create_metrics(0.08))
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_config.experimental_enabled = {}
            mock_get_config.return_value = mock_config
            
            # Run optimization with limited combinations (first 5)
            with patch("ml.optimizer._get_experimental_feature_combinations") as mock_combos:
                combos = _get_experimental_feature_combinations(include_derived=False)
                mock_combos.return_value = combos[:5]  # Use first 5 combinations
                
                progress = run_optimization(
                    train_single_step_fn=mock_train_single,
                    train_two_step_fn=mock_train_two_step,
                    build_dataset_fn=mock_build_dataset,
                    min_samples=50,
                    max_workers=2,  # Use 2 parallel workers
                )
        
        # Should have 5 combinations × 2 models = 10 results
        assert len(progress.results) == 10, f"Expected 10 results, got {len(progress.results)}"
        assert progress.phase == "complete"
        
        # All results should be successful
        assert all(r.success for r in progress.results), "Some trainings failed"
        
        # Each result should have its own feature configuration
        feature_configs = [frozenset(k for k, v in r.experimental_features.items() if v) 
                          for r in progress.results]
        
        # We should have distinct configurations (accounting for 2 models per config)
        unique_configs = set(feature_configs)
        assert len(unique_configs) == 5, f"Expected 5 unique configs, got {len(unique_configs)}"
