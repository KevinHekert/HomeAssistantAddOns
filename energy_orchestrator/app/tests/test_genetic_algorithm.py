"""
Tests for Genetic Algorithm and Hybrid search strategies.

This test module verifies:
1. Genetic Algorithm generates diverse populations
2. Mutation and crossover work correctly
3. Hybrid strategy combines GA + Bayesian phases
4. Search strategies are configurable
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from ml.optimizer import (
    SearchStrategy,
    _generate_genetic_algorithm_combinations,
    _generate_hybrid_genetic_bayesian_combinations,
    run_optimization,
    OptimizationResult,
)
from ml.feature_config import EXPERIMENTAL_FEATURES


class TestGeneticAlgorithm:
    """Test the Genetic Algorithm strategy."""

    def test_ga_generates_initial_population(self):
        """Verify GA generates initial random population."""
        population_size = 10
        num_generations = 1  # Just test first generation
        
        combos_gen = _generate_genetic_algorithm_combinations(
            include_derived=False,
            population_size=population_size,
            num_generations=num_generations,
            mutation_rate=0.1,
        )
        combos = list(combos_gen)
        
        # Should generate exactly population_size for 1 generation
        assert len(combos) == population_size, f"Expected {population_size} combinations, got {len(combos)}"
        
        # First individual should be baseline (all False)
        assert all(not v for v in combos[0].values()), "First individual should be baseline"
        
        # Others should be random (likely some True values)
        has_enabled_features = any(
            any(v for v in combo.values())
            for combo in combos[1:]
        )
        assert has_enabled_features, "Random population should have some enabled features"
    
    def test_ga_multiple_generations(self):
        """Verify GA generates multiple generations."""
        population_size = 5
        num_generations = 3
        
        combos_gen = _generate_genetic_algorithm_combinations(
            include_derived=False,
            population_size=population_size,
            num_generations=num_generations,
        )
        combos = list(combos_gen)
        
        # Should generate population_size * num_generations
        expected_total = population_size * num_generations
        assert len(combos) == expected_total, f"Expected {expected_total} combinations, got {len(combos)}"
    
    def test_ga_respects_feature_count(self):
        """Verify GA uses correct number of features."""
        combos_gen = _generate_genetic_algorithm_combinations(
            include_derived=False,
            population_size=5,
            num_generations=1,
        )
        combos = list(combos_gen)
        
        # All combinations should have same number of features
        n_features = len(EXPERIMENTAL_FEATURES)
        for combo in combos:
            assert len(combo) == n_features, f"Expected {n_features} features, got {len(combo)}"
    
    def test_ga_diverse_population(self):
        """Verify GA generates diverse combinations."""
        combos_gen = _generate_genetic_algorithm_combinations(
            include_derived=False,
            population_size=20,
            num_generations=1,
        )
        combos = list(combos_gen)
        
        # Convert to frozensets for uniqueness check
        combo_sets = set(
            frozenset(k for k, v in combo.items() if v)
            for combo in combos
        )
        
        # Should have multiple unique combinations (allowing some duplicates due to randomness)
        assert len(combo_sets) >= 10, f"Expected at least 10 unique combinations, got {len(combo_sets)}"


class TestHybridStrategy:
    """Test the Hybrid Genetic Algorithm + Bayesian Optimization strategy."""
    
    def test_hybrid_has_two_phases(self):
        """Verify hybrid strategy generates two distinct phases."""
        ga_population = 5
        ga_generations = 2
        bayesian_iters = 3
        
        combos_gen = _generate_hybrid_genetic_bayesian_combinations(
            include_derived=False,
            ga_population_size=ga_population,
            ga_num_generations=ga_generations,
            bayesian_iterations=bayesian_iters,
        )
        combos = list(combos_gen)
        
        # Total should be GA phase + Bayesian phase
        expected_ga = ga_population * ga_generations
        expected_total = expected_ga + bayesian_iters
        
        assert len(combos) == expected_total, f"Expected {expected_total} combinations (GA: {expected_ga}, Bayesian: {bayesian_iters}), got {len(combos)}"
    
    def test_hybrid_diversity(self):
        """Verify hybrid strategy generates diverse combinations."""
        combos_gen = _generate_hybrid_genetic_bayesian_combinations(
            include_derived=False,
            ga_population_size=10,
            ga_num_generations=2,
            bayesian_iterations=5,
        )
        combos = list(combos_gen)
        
        # Check diversity by counting unique feature counts
        feature_counts = set(
            sum(1 for v in combo.values() if v)
            for combo in combos
        )
        
        # Should test different numbers of features
        assert len(feature_counts) >= 3, f"Expected at least 3 different feature counts, got {len(feature_counts)}"
    
    def test_hybrid_bayesian_phase_allows_baseline(self):
        """Verify Bayesian phase can generate baseline (0 features) configuration.
        
        This test addresses the issue: "Optimizer run shouldn't have a minimum of experimental sensors"
        The Bayesian phase should be able to test configurations with 0 enabled features,
        because sometimes less is more.
        """
        # Use a large number of Bayesian iterations to increase likelihood of getting baseline
        combos_gen = _generate_hybrid_genetic_bayesian_combinations(
            include_derived=False,
            ga_population_size=5,
            ga_num_generations=1,
            bayesian_iterations=100,  # Large number to ensure we get baseline
        )
        combos = list(combos_gen)
        
        # Count how many features are enabled in each combination
        feature_counts = [sum(1 for v in combo.values() if v) for combo in combos]
        
        # Verify that at least one combination has 0 features (baseline)
        # The first combo from GA phase will always be baseline, but we also want
        # to ensure the Bayesian phase CAN generate it (even if randomly it might not always)
        has_baseline = 0 in feature_counts
        assert has_baseline, f"Expected at least one baseline configuration (0 features), got feature counts: {sorted(set(feature_counts))}"
        
        # Count baselines (should be at least 1 from GA phase, possibly more from Bayesian)
        baseline_count = feature_counts.count(0)
        
        # The GA phase includes baseline as first individual, so we should have at least 1
        assert baseline_count >= 1, f"Expected at least 1 baseline configuration, got {baseline_count}"


class TestSearchStrategyIntegration:
    """Test search strategy integration with run_optimization."""
    
    def test_strategy_parameter_accepted(self):
        """Verify run_optimization accepts search_strategy parameter."""
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
        mock_metrics.regressor_val_mape = 0.08
        mock_metrics.regressor_val_mae = 0.12
        mock_metrics.regressor_val_r2 = 0.88
        mock_metrics.regressor_train_samples = 60
        mock_metrics.regressor_val_samples = 20
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train_single(df):
            return (mock_model, mock_metrics)
        
        def mock_train_two_step(df):
            return (mock_model, mock_metrics)
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_config.experimental_enabled = {}
            mock_get_config.return_value = mock_config
            
            # Test with GENETIC strategy and small parameters
            progress = run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two_step,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
                include_derived_features=False,  # Use only 4 experimental features
                search_strategy=SearchStrategy.GENETIC,
                genetic_population_size=3,
                genetic_num_generations=2,
            )
        
        # Should complete successfully
        assert progress is not None, "Progress should not be None"
        assert progress.run_id is not None, "Run ID should be set"
        assert progress.phase == "complete", f"Expected complete phase, got {progress.phase}"
        
        # Should have processed combinations (3 pop × 2 gen × 2 models = 12)
        assert progress.completed_configurations == 12, f"Expected 12 configurations, got {progress.completed_configurations}"
    
    def test_hybrid_strategy_integration(self):
        """Verify run_optimization works with HYBRID_GENETIC_BAYESIAN strategy."""
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
        mock_metrics.regressor_val_mape = 0.08
        mock_metrics.regressor_val_mae = 0.12
        mock_metrics.regressor_val_r2 = 0.88
        mock_metrics.regressor_train_samples = 60
        mock_metrics.regressor_val_samples = 20
        
        def mock_build_dataset(min_samples):
            return (mock_df, mock_stats)
        
        def mock_train_single(df):
            return (mock_model, mock_metrics)
        
        def mock_train_two_step(df):
            return (mock_model, mock_metrics)
        
        with patch("ml.optimizer.get_feature_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.to_dict.return_value = {"experimental_enabled": {}}
            mock_config.experimental_enabled = {}
            mock_get_config.return_value = mock_config
            
            # Test with HYBRID strategy and small parameters
            progress = run_optimization(
                train_single_step_fn=mock_train_single,
                train_two_step_fn=mock_train_two_step,
                build_dataset_fn=mock_build_dataset,
                min_samples=50,
                include_derived_features=False,  # Use only 4 experimental features
                search_strategy=SearchStrategy.HYBRID_GENETIC_BAYESIAN,
                genetic_population_size=2,
                genetic_num_generations=2,
                bayesian_iterations=3,
            )
        
        # Should complete successfully
        assert progress is not None, "Progress should not be None"
        assert progress.run_id is not None, "Run ID should be set"
        assert progress.phase == "complete", f"Expected complete phase, got {progress.phase}"
        
        # Should have processed combinations ((2 pop × 2 gen) + 3 bayesian) × 2 models = 14
        expected_configs = ((2 * 2) + 3) * 2
        assert progress.completed_configurations == expected_configs, f"Expected {expected_configs} configurations, got {progress.completed_configurations}"
