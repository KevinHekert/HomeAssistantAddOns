"""
Settings optimizer for finding the best model configuration.

This module provides:
- Systematic search through different feature configurations
- Training both single-step and two-step models with each configuration
- Comparison based on Val MAPE (%)
- Saving/restoring original settings

The optimizer cycles through combinations of experimental features
and time window options to find the configuration that produces the
lowest validation MAPE.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
import copy

from ml.feature_config import (
    FeatureConfiguration,
    EXPERIMENTAL_FEATURES,
    get_feature_config,
    reload_feature_config,
)

_Logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a single model training configuration."""
    config_name: str
    model_type: str  # "single_step" or "two_step"
    experimental_features: dict[str, bool]
    val_mape_pct: Optional[float]
    val_mae_kwh: Optional[float]
    val_r2: Optional[float]
    train_samples: int
    val_samples: int
    success: bool
    error_message: Optional[str] = None
    training_timestamp: Optional[datetime] = None


@dataclass
class OptimizerProgress:
    """Progress tracking for the optimization process."""
    total_configurations: int
    completed_configurations: int
    current_configuration: str
    current_model_type: str
    phase: str  # "initializing", "training", "complete", "error"
    log_messages: list[str] = field(default_factory=list)
    results: list[OptimizationResult] = field(default_factory=list)
    best_result: Optional[OptimizationResult] = None
    original_settings: Optional[dict] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


def _get_experimental_feature_combinations() -> list[dict[str, bool]]:
    """
    Generate all combinations of experimental features to test.
    
    Rather than testing all 2^N combinations (which could be huge),
    we test:
    1. All features disabled (baseline)
    2. Each feature enabled individually
    3. Logical groups of features together
    
    Returns:
        List of experimental feature state dictionaries
    """
    feature_names = [f.name for f in EXPERIMENTAL_FEATURES]
    
    combinations = []
    
    # 1. Baseline: all experimental features disabled
    combinations.append({name: False for name in feature_names})
    
    # 2. Each feature enabled individually
    for name in feature_names:
        config = {n: False for n in feature_names}
        config[name] = True
        combinations.append(config)
    
    # 3. Logical groups: time-related features together
    time_features = ["day_of_week", "is_weekend", "is_night"]
    if all(f in feature_names for f in time_features):
        config = {name: False for name in feature_names}
        for tf in time_features:
            config[tf] = True
        combinations.append(config)
    
    # 4. All weather aggregations
    weather_agg_features = ["pressure", "outdoor_temp_avg_6h", "outdoor_temp_avg_7d"]
    if all(f in feature_names for f in weather_agg_features):
        config = {name: False for name in feature_names}
        for wf in weather_agg_features:
            config[wf] = True
        combinations.append(config)
    
    # 5. Heating-related features
    heating_features = ["heating_kwh_last_7d", "heating_degree_hours_24h", "heating_degree_hours_7d"]
    if all(f in feature_names for f in heating_features):
        config = {name: False for name in feature_names}
        for hf in heating_features:
            config[hf] = True
        combinations.append(config)
    
    # 6. All experimental features enabled
    combinations.append({name: True for name in feature_names})
    
    return combinations


def _configuration_to_name(experimental_enabled: dict[str, bool]) -> str:
    """Convert a feature configuration to a human-readable name."""
    enabled = [name for name, enabled in experimental_enabled.items() if enabled]
    if not enabled:
        return "Baseline (core features only)"
    elif len(enabled) == len(experimental_enabled):
        return "All features enabled"
    elif len(enabled) <= 3:
        return "+" + ", +".join(enabled)
    else:
        return f"+{len(enabled)} experimental features"


def run_optimization(
    train_single_step_fn: Callable,
    train_two_step_fn: Callable,
    build_dataset_fn: Callable,
    progress_callback: Optional[Callable[[OptimizerProgress], None]] = None,
    min_samples: int = 50,
) -> OptimizerProgress:
    """
    Run the settings optimizer to find the best configuration.
    
    This function:
    1. Saves current settings
    2. Iterates through feature configurations
    3. Trains both single-step and two-step models
    4. Compares Val MAPE to find the best configuration
    5. Reports progress via callback
    
    Args:
        train_single_step_fn: Function to train single-step model (df) -> (model, metrics)
        train_two_step_fn: Function to train two-step model (df) -> (model, metrics)
        build_dataset_fn: Function to build feature dataset (min_samples) -> (df, stats)
        progress_callback: Optional callback for progress updates
        min_samples: Minimum samples required for training
        
    Returns:
        OptimizerProgress with all results and the best configuration
    """
    # Initialize progress
    combinations = _get_experimental_feature_combinations()
    # Each combination is tested with both models (2 trainings per configuration)
    total_configs = len(combinations) * 2
    
    progress = OptimizerProgress(
        total_configurations=total_configs,
        completed_configurations=0,
        current_configuration="",
        current_model_type="",
        phase="initializing",
        start_time=datetime.now(),
    )
    
    # Save original settings
    original_config = get_feature_config()
    progress.original_settings = original_config.to_dict()
    progress.log_messages.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Optimizer started"
    )
    progress.log_messages.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Original settings saved"
    )
    progress.log_messages.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Testing {len(combinations)} configurations with 2 models each ({total_configs} total trainings)"
    )
    
    if progress_callback:
        progress_callback(progress)
    
    try:
        for combo in combinations:
            config_name = _configuration_to_name(combo)
            
            # Apply configuration
            progress.current_configuration = config_name
            progress.phase = "training"
            
            progress.log_messages.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Testing configuration: {config_name}"
            )
            
            # Update feature config
            config = get_feature_config()
            for feature_name, enabled in combo.items():
                if enabled:
                    config.enable_experimental_feature(feature_name)
                else:
                    config.disable_experimental_feature(feature_name)
            # Don't save to disk - we just want to test in memory
            
            # Build dataset with current configuration
            try:
                df, stats = build_dataset_fn(min_samples=min_samples)
                
                if df is None:
                    progress.log_messages.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Skipping {config_name}: insufficient data"
                    )
                    # Record failed results for both model types
                    for model_type in ["single_step", "two_step"]:
                        result = OptimizationResult(
                            config_name=config_name,
                            model_type=model_type,
                            experimental_features=combo.copy(),
                            val_mape_pct=None,
                            val_mae_kwh=None,
                            val_r2=None,
                            train_samples=0,
                            val_samples=0,
                            success=False,
                            error_message="Insufficient data for training",
                        )
                        progress.results.append(result)
                        progress.completed_configurations += 1
                    if progress_callback:
                        progress_callback(progress)
                    continue
                    
            except Exception as e:
                _Logger.error("Error building dataset for %s: %s", config_name, e)
                progress.log_messages.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Error building dataset: {str(e)}"
                )
                for model_type in ["single_step", "two_step"]:
                    result = OptimizationResult(
                        config_name=config_name,
                        model_type=model_type,
                        experimental_features=combo.copy(),
                        val_mape_pct=None,
                        val_mae_kwh=None,
                        val_r2=None,
                        train_samples=0,
                        val_samples=0,
                        success=False,
                        error_message=str(e),
                    )
                    progress.results.append(result)
                    progress.completed_configurations += 1
                if progress_callback:
                    progress_callback(progress)
                continue
            
            # Train single-step model
            progress.current_model_type = "single_step"
            progress.log_messages.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Training single-step model..."
            )
            if progress_callback:
                progress_callback(progress)
            
            try:
                model, metrics = train_single_step_fn(df)
                result = OptimizationResult(
                    config_name=config_name,
                    model_type="single_step",
                    experimental_features=combo.copy(),
                    val_mape_pct=metrics.val_mape * 100 if metrics.val_mape == metrics.val_mape else None,
                    val_mae_kwh=metrics.val_mae,
                    val_r2=metrics.val_r2,
                    train_samples=metrics.train_samples,
                    val_samples=metrics.val_samples,
                    success=True,
                    training_timestamp=datetime.now(),
                )
                mape_str = f"{result.val_mape_pct:.2f}%" if result.val_mape_pct is not None else "N/A"
                progress.log_messages.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Single-step: Val MAPE = {mape_str}"
                )
            except Exception as e:
                _Logger.error("Error training single-step model: %s", e)
                result = OptimizationResult(
                    config_name=config_name,
                    model_type="single_step",
                    experimental_features=combo.copy(),
                    val_mape_pct=None,
                    val_mae_kwh=None,
                    val_r2=None,
                    train_samples=0,
                    val_samples=0,
                    success=False,
                    error_message=str(e),
                )
                progress.log_messages.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Single-step training failed: {str(e)}"
                )
            
            progress.results.append(result)
            progress.completed_configurations += 1
            
            # Update best result if this is better
            if result.success and result.val_mape_pct is not None:
                if (
                    progress.best_result is None
                    or progress.best_result.val_mape_pct is None
                    or result.val_mape_pct < progress.best_result.val_mape_pct
                ):
                    progress.best_result = result
            
            if progress_callback:
                progress_callback(progress)
            
            # Train two-step model
            progress.current_model_type = "two_step"
            progress.log_messages.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Training two-step model..."
            )
            if progress_callback:
                progress_callback(progress)
            
            try:
                model, metrics = train_two_step_fn(df)
                # Two-step model uses regressor MAPE for comparison
                result = OptimizationResult(
                    config_name=config_name,
                    model_type="two_step",
                    experimental_features=combo.copy(),
                    val_mape_pct=metrics.regressor_val_mape * 100 if metrics.regressor_val_mape == metrics.regressor_val_mape else None,
                    val_mae_kwh=metrics.regressor_val_mae,
                    val_r2=metrics.regressor_val_r2,
                    train_samples=metrics.regressor_train_samples,
                    val_samples=metrics.regressor_val_samples,
                    success=True,
                    training_timestamp=datetime.now(),
                )
                mape_str = f"{result.val_mape_pct:.2f}%" if result.val_mape_pct is not None else "N/A"
                progress.log_messages.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Two-step: Val MAPE = {mape_str}"
                )
            except Exception as e:
                _Logger.error("Error training two-step model: %s", e)
                result = OptimizationResult(
                    config_name=config_name,
                    model_type="two_step",
                    experimental_features=combo.copy(),
                    val_mape_pct=None,
                    val_mae_kwh=None,
                    val_r2=None,
                    train_samples=0,
                    val_samples=0,
                    success=False,
                    error_message=str(e),
                )
                progress.log_messages.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Two-step training failed: {str(e)}"
                )
            
            progress.results.append(result)
            progress.completed_configurations += 1
            
            # Update best result if this is better
            if result.success and result.val_mape_pct is not None:
                if (
                    progress.best_result is None
                    or progress.best_result.val_mape_pct is None
                    or result.val_mape_pct < progress.best_result.val_mape_pct
                ):
                    progress.best_result = result
            
            if progress_callback:
                progress_callback(progress)
        
        # Optimization complete
        progress.phase = "complete"
        progress.end_time = datetime.now()
        
        if progress.best_result:
            progress.log_messages.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Optimization complete!"
            )
            progress.log_messages.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Best configuration: {progress.best_result.config_name}"
            )
            progress.log_messages.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Best model: {progress.best_result.model_type}"
            )
            mape_str = f"{progress.best_result.val_mape_pct:.2f}%" if progress.best_result.val_mape_pct else "N/A"
            progress.log_messages.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Best Val MAPE: {mape_str}"
            )
        else:
            progress.log_messages.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Optimization complete (no valid results)"
            )
        
    except Exception as e:
        _Logger.error("Optimizer error: %s", e, exc_info=True)
        progress.phase = "error"
        progress.error_message = str(e)
        progress.log_messages.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}"
        )
        progress.end_time = datetime.now()
    
    finally:
        # Restore original settings
        if progress.original_settings:
            config = get_feature_config()
            original = progress.original_settings
            for feature_name, enabled in original.get("experimental_enabled", {}).items():
                if enabled:
                    config.enable_experimental_feature(feature_name)
                else:
                    config.disable_experimental_feature(feature_name)
            # Note: we don't save to disk here - user can choose to apply best settings
            progress.log_messages.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Original settings restored"
            )
    
    if progress_callback:
        progress_callback(progress)
    
    return progress


def apply_best_configuration(
    best_result: OptimizationResult,
    enable_two_step: bool = False,
) -> bool:
    """
    Apply the best configuration found by the optimizer.
    
    Args:
        best_result: The best OptimizationResult from optimization
        enable_two_step: If True, also enable two-step prediction mode
        
    Returns:
        True if successfully applied and saved
    """
    try:
        config = get_feature_config()
        
        # Apply experimental feature settings
        for feature_name, enabled in best_result.experimental_features.items():
            if enabled:
                config.enable_experimental_feature(feature_name)
            else:
                config.disable_experimental_feature(feature_name)
        
        # Optionally enable two-step prediction
        if enable_two_step and best_result.model_type == "two_step":
            config.enable_two_step_prediction()
        elif best_result.model_type == "single_step":
            config.disable_two_step_prediction()
        
        # Save to disk
        return config.save()
        
    except Exception as e:
        _Logger.error("Error applying best configuration: %s", e)
        return False
