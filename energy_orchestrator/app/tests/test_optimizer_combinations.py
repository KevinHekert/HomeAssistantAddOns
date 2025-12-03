"""
Tests for optimizer combination generation.

This test module verifies that:
1. The optimizer generates the expected number and types of combinations
2. Specific combinations mentioned in the issue are included
3. Each parallel run uses different feature settings (thread safety)
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from collections import Counter

from ml.optimizer import (
    _get_experimental_feature_combinations,
    run_optimization,
    _train_single_configuration,
)
from ml.feature_config import EXPERIMENTAL_FEATURES


class TestCombinationGeneration:
    """Test the combination generation logic."""

    def test_combination_count_includes_pairwise(self):
        """Verify that pairwise (2-feature) combinations are generated."""
        combos = _get_experimental_feature_combinations(include_derived=False)
        
        # Count combinations by number of enabled features
        enabled_counts = Counter()
        for combo in combos:
            count = sum(1 for v in combo.values() if v)
            enabled_counts[count] += 1
        
        # Should have:
        # - 1 baseline (0 features)
        # - 10 individual features
        # - 45 pairwise combinations (10 choose 2)
        # - 3 logical groups (3 features each)
        # - 1 all-enabled
        assert enabled_counts[0] == 1, "Should have 1 baseline"
        assert enabled_counts[1] == 10, "Should have 10 individual features"
        assert enabled_counts[2] == 45, "Should have 45 pairwise combinations (10 choose 2)"
        assert enabled_counts[3] == 3, "Should have 3 logical groups"
        assert enabled_counts[10] == 1, "Should have 1 all-enabled"
    
    def test_total_combination_count(self):
        """Verify the total number of combinations."""
        combos = _get_experimental_feature_combinations(include_derived=False)
        
        # 1 baseline + 10 individual + 45 pairwise + 3 groups + 1 all-enabled = 60
        assert len(combos) == 60, f"Expected 60 combinations, got {len(combos)}"
    
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
    
    def test_logical_groups_exist(self):
        """Verify logical groups of 3+ features exist."""
        combos = _get_experimental_feature_combinations(include_derived=False)
        
        # Time features group
        time_group = {'day_of_week', 'is_weekend', 'is_night'}
        found_time = any(
            {k for k, v in combo.items() if v} == time_group
            for combo in combos
        )
        assert found_time, "Time features group not found"
        
        # Weather features group
        weather_group = {'pressure', 'outdoor_temp_avg_6h', 'outdoor_temp_avg_7d'}
        found_weather = any(
            {k for k, v in combo.items() if v} == weather_group
            for combo in combos
        )
        assert found_weather, "Weather features group not found"
        
        # Heating features group
        heating_group = {'heating_kwh_last_7d', 'heating_degree_hours_24h', 'heating_degree_hours_7d'}
        found_heating = any(
            {k for k, v in combo.items() if v} == heating_group
            for combo in combos
        )
        assert found_heating, "Heating features group not found"
    
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
        combo_sets = []
        for combo in combos:
            enabled = frozenset(k for k, v in combo.items() if v)
            combo_sets.append(enabled)
        
        # Check for duplicates
        assert len(combo_sets) == len(set(combo_sets)), "Duplicate combinations found"


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
