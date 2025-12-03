"""
Settings optimizer for finding the best model configuration.

This module provides:
- Systematic search through different feature configurations
- Training both single-step and two-step models with each configuration
- Comparison based on Val MAPE (%)
- Saving/restoring original settings
- Parallel training with configurable worker count

The optimizer cycles through combinations of experimental features
and time window options to find the configuration that produces the
lowest validation MAPE. It supports parallel execution for faster
optimization of multiple feature combinations.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ml.feature_config import (
    FeatureConfiguration,
    EXPERIMENTAL_FEATURES,
    get_feature_config,
    reload_feature_config,
)

_Logger = logging.getLogger(__name__)

# Lock for thread-safe progress updates
_progress_lock = threading.Lock()

# Lock for thread-safe feature configuration modifications
# This ensures that feature config changes and dataset building happen atomically
_config_lock = threading.Lock()

# Minimum number of features required to create a logical group combination
MIN_FEATURES_FOR_GROUP = 2


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


def _get_all_available_features() -> list[str]:
    """
    Get all available features including experimental and derived features.
    
    This includes:
    - All experimental features from EXPERIMENTAL_FEATURES
    - All derived features currently defined in feature configuration
    
    Returns:
        List of all available feature names
    """
    config = get_feature_config()
    
    # Start with experimental features
    feature_names = [f.name for f in EXPERIMENTAL_FEATURES]
    
    # Add any derived features from experimental_enabled that aren't in EXPERIMENTAL_FEATURES
    for feature_name, enabled in config.experimental_enabled.items():
        if feature_name not in feature_names:
            feature_names.append(feature_name)
    
    _Logger.info("Found %d total features for optimization (experimental + derived)", len(feature_names))
    return feature_names


def _get_experimental_feature_combinations(include_derived: bool = True) -> list[dict[str, bool]]:
    """
    Generate all combinations of experimental features to test.
    
    Rather than testing all 2^N combinations (which could be huge),
    we test:
    1. All features disabled (baseline)
    2. Each feature enabled individually
    3. Logical groups of features together
    
    Args:
        include_derived: If True, includes derived features in combinations
    
    Returns:
        List of experimental feature state dictionaries
    """
    if include_derived:
        feature_names = _get_all_available_features()
    else:
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
    matching_time = [f for f in feature_names if f in time_features]
    if len(matching_time) >= MIN_FEATURES_FOR_GROUP:
        config = {name: False for name in feature_names}
        for tf in matching_time:
            config[tf] = True
        combinations.append(config)
    
    # 4. All weather aggregations
    weather_agg_features = ["pressure", "outdoor_temp_avg_6h", "outdoor_temp_avg_7d"]
    matching_weather = [f for f in feature_names if f in weather_agg_features]
    if len(matching_weather) >= MIN_FEATURES_FOR_GROUP:
        config = {name: False for name in feature_names}
        for wf in matching_weather:
            config[wf] = True
        combinations.append(config)
    
    # 5. Heating-related features
    heating_features = ["heating_kwh_last_7d", "heating_degree_hours_24h", "heating_degree_hours_7d"]
    matching_heating = [f for f in feature_names if f in heating_features]
    if len(matching_heating) >= MIN_FEATURES_FOR_GROUP:
        config = {name: False for name in feature_names}
        for hf in matching_heating:
            config[hf] = True
        combinations.append(config)
    
    # 6. All experimental features enabled
    combinations.append({name: True for name in feature_names})
    
    _Logger.info("Generated %d feature combinations to test", len(combinations))
    return combinations


def _train_single_configuration(
    config_name: str,
    combo: dict[str, bool],
    model_type: str,
    train_fn: Callable,
    build_dataset_fn: Callable,
    min_samples: int,
) -> OptimizationResult:
    """
    Train a single model configuration.
    
    This function is designed to be run in parallel by multiple workers.
    
    Args:
        config_name: Human-readable name for this configuration
        combo: Feature enable/disable dictionary
        model_type: "single_step" or "two_step"
        train_fn: Training function to use
        build_dataset_fn: Function to build feature dataset
        min_samples: Minimum samples required
        
    Returns:
        OptimizationResult with training metrics
    """
    try:
        # Use lock to ensure feature configuration and dataset building are atomic
        # This prevents race conditions where Thread A's config could be overwritten
        # by Thread B before Thread A finishes building its dataset
        with _config_lock:
            config = get_feature_config()
            # Apply this thread's feature configuration
            for feature_name, enabled in combo.items():
                if enabled:
                    config.enable_feature(feature_name)
                else:
                    config.disable_feature(feature_name)
            
            # Build dataset with current configuration while holding the lock
            # This ensures the config is consistent throughout dataset building
            df, stats = build_dataset_fn(min_samples=min_samples)
        
        if df is None:
            return OptimizationResult(
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
        
        # Train model
        model, metrics = train_fn(df)
        
        # Extract metrics based on model type
        if model_type == "single_step":
            val_mape_pct = None
            if metrics.val_mape is not None and not math.isnan(metrics.val_mape):
                val_mape_pct = metrics.val_mape * 100
            
            return OptimizationResult(
                config_name=config_name,
                model_type=model_type,
                experimental_features=combo.copy(),
                val_mape_pct=val_mape_pct,
                val_mae_kwh=metrics.val_mae,
                val_r2=metrics.val_r2,
                train_samples=metrics.train_samples,
                val_samples=metrics.val_samples,
                success=True,
                training_timestamp=datetime.now(),
            )
        else:  # two_step
            val_mape_pct = None
            if metrics.regressor_val_mape is not None and not math.isnan(metrics.regressor_val_mape):
                val_mape_pct = metrics.regressor_val_mape * 100
            
            return OptimizationResult(
                config_name=config_name,
                model_type=model_type,
                experimental_features=combo.copy(),
                val_mape_pct=val_mape_pct,
                val_mae_kwh=metrics.regressor_val_mae,
                val_r2=metrics.regressor_val_r2,
                train_samples=metrics.regressor_train_samples,
                val_samples=metrics.regressor_val_samples,
                success=True,
                training_timestamp=datetime.now(),
            )
            
    except Exception as e:
        _Logger.error("Error training %s model for %s: %s", model_type, config_name, e)
        return OptimizationResult(
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
    max_workers: int = 3,
    include_derived_features: bool = True,
) -> OptimizerProgress:
    """
    Run the settings optimizer to find the best configuration.
    
    This function:
    1. Saves current settings
    2. Iterates through feature configurations
    3. Trains both single-step and two-step models (in parallel)
    4. Compares Val MAPE to find the best configuration
    5. Reports progress via callback
    
    Args:
        train_single_step_fn: Function to train single-step model (df) -> (model, metrics)
        train_two_step_fn: Function to train two-step model (df) -> (model, metrics)
        build_dataset_fn: Function to build feature dataset (min_samples) -> (df, stats)
        progress_callback: Optional callback for progress updates
        min_samples: Minimum samples required for training
        max_workers: Maximum number of parallel workers (default: 3)
        include_derived_features: Whether to include derived features in combinations (default: True)
        
    Returns:
        OptimizerProgress with all results and the best configuration
    """
    # Initialize progress
    combinations = _get_experimental_feature_combinations(include_derived=include_derived_features)
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
    
    progress.log_messages.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Using {max_workers} parallel workers for training"
    )
    
    if progress_callback:
        progress_callback(progress)
    
    try:
        progress.phase = "training"
        
        # Create a list of all training tasks (config + model_type combinations)
        training_tasks = []
        for combo in combinations:
            config_name = _configuration_to_name(combo)
            training_tasks.append((config_name, combo, "single_step", train_single_step_fn))
            training_tasks.append((config_name, combo, "two_step", train_two_step_fn))
        
        # Use ThreadPoolExecutor for parallel training
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all training tasks
            future_to_task = {}
            for config_name, combo, model_type, train_fn in training_tasks:
                future = executor.submit(
                    _train_single_configuration,
                    config_name,
                    combo,
                    model_type,
                    train_fn,
                    build_dataset_fn,
                    min_samples,
                )
                future_to_task[future] = (config_name, model_type)
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_task):
                config_name, model_type = future_to_task[future]
                
                try:
                    result = future.result()
                    
                    # Thread-safe progress update
                    with _progress_lock:
                        progress.results.append(result)
                        progress.completed_configurations += 1
                        progress.current_configuration = config_name
                        progress.current_model_type = model_type
                        
                        # Log the result
                        if result.success:
                            mape_str = f"{result.val_mape_pct:.2f}%" if result.val_mape_pct is not None else "N/A"
                            progress.log_messages.append(
                                f"[{datetime.now().strftime('%H:%M:%S')}] [{progress.completed_configurations}/{total_configs}] {config_name} ({model_type}): Val MAPE = {mape_str}"
                            )
                        else:
                            progress.log_messages.append(
                                f"[{datetime.now().strftime('%H:%M:%S')}] [{progress.completed_configurations}/{total_configs}] {config_name} ({model_type}): Failed - {result.error_message}"
                            )
                        
                        # Update best result if this is better
                        if result.success and result.val_mape_pct is not None:
                            if (
                                progress.best_result is None
                                or progress.best_result.val_mape_pct is None
                                or result.val_mape_pct < progress.best_result.val_mape_pct
                            ):
                                progress.best_result = result
                                progress.log_messages.append(
                                    f"[{datetime.now().strftime('%H:%M:%S')}] 🏆 New best: {config_name} ({model_type}) with Val MAPE = {result.val_mape_pct:.2f}%"
                                )
                    
                    # Call progress callback outside the lock to avoid deadlocks
                    if progress_callback:
                        progress_callback(progress)
                        
                except Exception as e:
                    _Logger.error("Error processing result for %s (%s): %s", config_name, model_type, e)
                    with _progress_lock:
                        progress.log_messages.append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] Error processing result: {str(e)}"
                        )
        
        # Optimization complete
        progress.phase = "complete"
        progress.end_time = datetime.now()
        
        with _progress_lock:
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
        with _progress_lock:
            progress.log_messages.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}"
            )
        progress.end_time = datetime.now()
    
    finally:
        # Restore original settings (handles both experimental and derived features)
        if progress.original_settings:
            config = get_feature_config()
            original = progress.original_settings
            for feature_name, enabled in original.get("experimental_enabled", {}).items():
                if enabled:
                    config.enable_feature(feature_name)
                else:
                    config.disable_feature(feature_name)
            # Note: we don't save to disk here - user can choose to apply best settings
            with _progress_lock:
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
    
    This function handles both experimental features and derived features.
    
    Args:
        best_result: The best OptimizationResult from optimization
        enable_two_step: If True, also enable two-step prediction mode
        
    Returns:
        True if successfully applied and saved
    """
    try:
        config = get_feature_config()
        
        # Apply feature settings (works for both experimental and derived features)
        for feature_name, enabled in best_result.experimental_features.items():
            if enabled:
                config.enable_feature(feature_name)
            else:
                config.disable_feature(feature_name)
        
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
