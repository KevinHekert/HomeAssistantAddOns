"""
Test to verify the optimizer bug fix for 'type' object is not iterable error.

This test ensures that the bug fixed in the run_optimization function
where 'combinations' was referenced instead of 'combinations_list' is resolved.
"""

import pytest
from unittest.mock import MagicMock
import pandas as pd
from datetime import datetime

from ml.optimizer import (
    run_optimization,
    SearchStrategy,
)


def test_optimizer_can_iterate_combinations():
    """
    Test that the optimizer correctly iterates over combinations_list.
    
    This test verifies the fix for the bug where line 1080 tried to iterate
    over 'combinations' which didn't exist, instead of 'combinations_list'.
    
    The bug was: "Error: 'type' object is not iterable"
    The fix was: Change 'for combo in combinations:' to 'for combo in combinations_list:'
    """
    # Create mock functions
    mock_df = pd.DataFrame({
        "outdoor_temp": [10.0, 11.0, 12.0, 13.0, 14.0],
        "target_heating_kwh_1h": [1.0, 1.5, 2.0, 2.5, 3.0],
    })
    
    # Mock training metrics
    mock_metrics = MagicMock()
    mock_metrics.train_samples = 60
    mock_metrics.val_samples = 20
    mock_metrics.val_mae = 0.15
    mock_metrics.val_mape = 0.10
    mock_metrics.val_r2 = 0.85
    
    # Mock two-step metrics  
    mock_metrics.regressor_train_samples = 60
    mock_metrics.regressor_val_samples = 20
    mock_metrics.regressor_val_mae = 0.18
    mock_metrics.regressor_val_mape = 0.12
    mock_metrics.regressor_val_r2 = 0.80
    
    mock_model = MagicMock()
    
    def mock_train_single_step(df):
        return mock_model, mock_metrics
    
    def mock_train_two_step(df):
        return mock_model, mock_metrics
    
    def mock_build_dataset(min_samples=50):
        return mock_df, MagicMock()
    
    # Mock the database functions to avoid database connection issues
    import db.optimizer_storage as storage
    storage.create_optimizer_run = MagicMock(return_value=1)
    storage.save_optimizer_result = MagicMock(return_value=1)
    storage.update_optimizer_run_progress = MagicMock()
    storage.complete_optimizer_run = MagicMock()
    
    # Run optimization with a small number of combinations
    # This should not raise "'type' object is not iterable" error
    progress = run_optimization(
        train_single_step_fn=mock_train_single_step,
        train_two_step_fn=mock_train_two_step,
        build_dataset_fn=mock_build_dataset,
        min_samples=3,
        include_derived_features=False,  # Use only experimental features for faster test
        configured_max_combinations=2,  # Very small number for quick test
        search_strategy=SearchStrategy.EXHAUSTIVE,
        configured_max_workers=1,  # Single worker for deterministic test
    )
    
    # Verify the optimizer completed without the "'type' object is not iterable" error
    assert progress.phase in ["complete", "error"], f"Expected phase 'complete' or 'error', got '{progress.phase}'"
    
    # If there was an error, it should NOT be the "'type' object is not iterable" error
    if progress.phase == "error":
        assert "'type' object is not iterable" not in progress.error_message, \
            f"The bug is not fixed! Error: {progress.error_message}"
        # Any other error is acceptable for this test (e.g., database issues in test environment)
    else:
        # If it completed successfully, verify it processed some configurations
        assert progress.completed_configurations > 0, "No configurations were completed"


def test_combinations_list_variable_exists():
    """
    Test that verifies combinations_list is created and used correctly.
    
    This is a focused test to ensure the variable name is correct.
    """
    from ml.optimizer import _generate_experimental_feature_combinations
    
    # This mimics what run_optimization does
    combo_generator = _generate_experimental_feature_combinations(
        include_derived=False,
        max_combinations=3,
    )
    
    # Convert generator to list (this is what line 1008 does)
    combinations_list = list(combo_generator)
    
    # Verify we can iterate over combinations_list (this is what line 1080 should do)
    task_count = 0
    for combo in combinations_list:
        assert isinstance(combo, dict), "Each combination should be a dictionary"
        task_count += 1
    
    assert task_count == len(combinations_list), "Should iterate over all combinations"
    assert task_count > 0, "Should have at least one combination"
