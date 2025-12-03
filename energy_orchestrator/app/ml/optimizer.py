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
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from itertools import combinations
import gc
import time
import psutil
import os

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


def _calculate_optimal_workers(max_memory_mb: Optional[float] = None) -> int:
    """
    Automatically calculate the optimal number of parallel workers based on system resources.
    
    This function considers:
    1. Available system memory
    2. Number of CPU cores
    3. Estimated memory per training task (~100-200 MB)
    4. User-defined memory limit (if provided)
    
    Args:
        max_memory_mb: Optional user-defined maximum memory in MB.
                       If None, uses 75% of available system memory.
    
    Returns:
        Optimal number of workers (minimum 1, maximum 10)
    """
    try:
        # Get system resources
        cpu_count = os.cpu_count() or 1
        sys_mem = psutil.virtual_memory()
        
        # Determine available memory for optimizer
        if max_memory_mb is not None:
            available_for_optimizer = max_memory_mb
        else:
            # Use 75% of available system memory as default
            total_mb = sys_mem.total / 1024 / 1024
            available_for_optimizer = total_mb * 0.75
        
        # Estimate memory per training task
        # Typical: 100-200 MB per task (DataFrame + model + overhead)
        # Conservative estimate: 200 MB
        estimated_memory_per_task = 200
        
        # Calculate max workers based on memory
        workers_by_memory = max(1, int(available_for_optimizer / estimated_memory_per_task))
        
        # Calculate max workers based on CPU (leave 1 core for system)
        workers_by_cpu = max(1, cpu_count - 1)
        
        # Take the minimum to avoid oversubscribing either resource
        optimal_workers = min(workers_by_memory, workers_by_cpu, 10)  # Cap at 10
        
        _Logger.info(
            "Auto-calculated optimal workers: %d (Memory: %.0f MB → %d workers, CPU: %d cores → %d workers)",
            optimal_workers, available_for_optimizer, workers_by_memory, cpu_count, workers_by_cpu
        )
        
        return optimal_workers
        
    except Exception as e:
        _Logger.warning("Failed to calculate optimal workers, defaulting to 1: %s", e)
        return 1


def _log_memory_usage(label: str) -> dict[str, float]:
    """
    Log current memory usage at INFO level.
    
    Args:
        label: Descriptive label for this memory check
        
    Returns:
        Dictionary with memory stats in MB
    """
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / 1024 / 1024  # Resident Set Size in MB
        vms_mb = mem_info.vms / 1024 / 1024  # Virtual Memory Size in MB
        
        # Get system-wide memory if available
        sys_mem = psutil.virtual_memory()
        available_mb = sys_mem.available / 1024 / 1024
        percent_used = sys_mem.percent
        
        _Logger.info(
            "%s - Memory: RSS=%.1f MB, VMS=%.1f MB, System Available=%.1f MB (%.1f%% used)",
            label, rss_mb, vms_mb, available_mb, percent_used
        )
        
        return {
            "rss_mb": rss_mb,
            "vms_mb": vms_mb,
            "available_mb": available_mb,
            "percent_used": percent_used,
        }
    except Exception as e:
        _Logger.warning("Failed to get memory usage: %s", e)
        return {}


def _should_allow_parallel_task(max_memory_mb: Optional[float] = None) -> bool:
    """
    Check if we have enough memory to start another parallel task.
    
    This function implements memory-based throttling to prevent OOM kills.
    If current memory usage exceeds the threshold, parallel tasks are throttled.
    
    Args:
        max_memory_mb: Maximum allowed memory in MB. If None, defaults to 1536 MB (75% of 2GB limit)
        
    Returns:
        True if we can safely start another parallel task, False otherwise
    """
    if max_memory_mb is None:
        max_memory_mb = 1536  # Default to 1.5GB (75% of typical 2GB container limit)
    
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / 1024 / 1024
        
        # Check if we're below the threshold
        if rss_mb < max_memory_mb:
            return True
        else:
            _Logger.debug(
                "Memory throttle active: RSS=%.1f MB exceeds threshold %.1f MB",
                rss_mb, max_memory_mb
            )
            return False
    except Exception as e:
        _Logger.warning("Failed to check memory for throttling: %s", e)
        # If we can't check memory, be conservative and don't allow parallel
        return False


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
    max_log_messages: int = 10  # Maximum number of log messages to keep
    
    def add_log_message(self, message: str) -> None:
        """
        Add a log message and maintain the tail limit.
        
        Keeps only the last max_log_messages entries to prevent memory issues
        with large optimization runs (1024+ combinations).
        
        Args:
            message: Log message to add
        """
        self.log_messages.append(message)
        
        # Keep only the last N messages
        if len(self.log_messages) > self.max_log_messages:
            self.log_messages = self.log_messages[-self.max_log_messages:]
    
    def get_top_results(self, n: int = 20) -> list[OptimizationResult]:
        """
        Get the top N results sorted by validation MAPE (lower is better).
        
        Args:
            n: Number of top results to return (default: 20)
            
        Returns:
            List of top N OptimizationResult objects sorted by Val MAPE
        """
        # Filter successful results with valid MAPE
        valid_results = [
            r for r in self.results 
            if r.success and r.val_mape_pct is not None
        ]
        
        # Sort by MAPE (ascending - lower is better)
        sorted_results = sorted(valid_results, key=lambda r: r.val_mape_pct)
        
        # Return top N
        return sorted_results[:n]



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


def _get_experimental_feature_combinations(
    include_derived: bool = True,
) -> list[dict[str, bool]]:
    """
    Generate ALL combinations of experimental features to test.
    
    This function generates all possible combinations (2^N):
    - Size 0: Baseline (all disabled)
    - Size 1: Each feature individually
    - Size 2: All 2-feature combinations
    - Size 3: All 3-feature combinations
    - ... up to all features
    - Size N: All features enabled
    
    For 10 features, this generates 2^10 = 1024 combinations.
    With 2 models tested per combination, that's 2048 total trainings.
    
    Args:
        include_derived: If True, includes derived features in combinations
    
    Returns:
        List of experimental feature state dictionaries (all 2^N combinations)
    """
    if include_derived:
        feature_names = _get_all_available_features()
    else:
        feature_names = [f.name for f in EXPERIMENTAL_FEATURES]
    
    n_features = len(feature_names)
    combos = []
    
    _Logger.info(
        "Generating ALL feature combinations (2^%d = %d combinations)",
        n_features,
        2 ** n_features,
    )
    
    # Generate all combinations for each size from 0 to n_features
    for size in range(n_features + 1):
        if size == 0:
            # Baseline: all features disabled
            combos.append({name: False for name in feature_names})
        else:
            # Generate all combinations of this size
            for feature_combo in combinations(feature_names, size):
                config = {name: False for name in feature_names}
                for feature_name in feature_combo:
                    config[feature_name] = True
                combos.append(config)
    
    _Logger.info("Generated %d feature combinations to test", len(combos))
    return combos


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
    
    Thread Safety:
        - Uses _config_lock to ensure atomic config modification and dataset building
        - Modifies the global feature config singleton (by design)
        - Lock ensures each thread's config state is consistent during dataset building
        - Training happens outside the lock and uses the built dataset
        - Original config is restored after all parallel tasks complete
    
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
        # by Thread B before Thread A finishes building its dataset.
        # Note: We intentionally mutate the global config (singleton pattern) and
        # serialize access to it. This is acceptable because:
        # 1. Parallel speedup comes from concurrent training, not dataset building
        # 2. Dataset building is typically fast compared to training
        # 3. Original config is restored after optimization completes
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
        
        # Explicitly delete DataFrame and model to free memory immediately
        # This is critical for long-running optimizations (2048 trainings)
        # to prevent memory accumulation and OOM kills
        del df
        del model
        
        # Force garbage collection to free memory immediately
        gc.collect()
        
        # Add a small delay to allow garbage collector to complete
        # This prevents memory accumulation during rapid sequential training
        time.sleep(0.5)
        
        # Extract metrics based on model type
        if model_type == "single_step":
            val_mape_pct = None
            if metrics.val_mape is not None and not math.isnan(metrics.val_mape):
                val_mape_pct = metrics.val_mape * 100
            
            result = OptimizationResult(
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
            
            result = OptimizationResult(
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
        
        return result
            
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
    include_derived_features: bool = True,
    max_memory_mb: Optional[float] = None,
) -> OptimizerProgress:
    """
    Run the settings optimizer to find the best configuration.
    
    This function:
    1. Saves current settings
    2. Automatically calculates optimal number of workers based on system resources
    3. Iterates through ALL feature combinations (2^N)
    4. Trains both single-step and two-step models with adaptive parallelism
    5. Compares Val MAPE to find the best configuration
    6. Reports progress via callback
    7. Logs memory usage at INFO level for monitoring
    
    For 10 experimental features, this tests 1024 combinations × 2 models = 2048 trainings.
    
    Memory Management:
    - Auto-calculates optimal workers based on available memory and CPU cores
    - Uses adaptive parallelism based on real-time memory availability
    - Throttles parallel workers when memory exceeds max_memory_mb threshold
    - Falls back to sequential processing when memory is constrained
    - Adds 0.5s delay after each training to allow garbage collection
    - Forces garbage collection every 10 iterations
    - Logs memory usage every 10 iterations at INFO level
    - Explicitly deletes DataFrames and models after each training
    
    Worker Calculation:
    - Automatically determines optimal workers from system resources
    - Considers: available memory, CPU cores, estimated task memory (~200MB)
    - Formula: min(memory_workers, cpu_workers, 10)
    - Example: 4GB RAM, 4 cores → min(20, 3, 10) = 3 workers
    
    Args:
        train_single_step_fn: Function to train single-step model (df) -> (model, metrics)
        train_two_step_fn: Function to train two-step model (df) -> (model, metrics)
        build_dataset_fn: Function to build feature dataset (min_samples) -> (df, stats)
        progress_callback: Optional callback for progress updates
        min_samples: Minimum samples required for training
        include_derived_features: Whether to include derived features in combinations (default: True)
        max_memory_mb: Maximum memory in MB before throttling parallel execution.
                       If None, defaults to 1536 MB (75% of 2GB limit).
                       Set via UI optimizer settings.
        
    Returns:
        OptimizerProgress with all results and the best configuration.
        Use progress.get_top_results(20) to get the top 20 results for UI display.
    """
    # Auto-calculate optimal number of workers based on system resources
    max_workers = _calculate_optimal_workers(max_memory_mb)
    
    # Initialize progress
    combinations = _get_experimental_feature_combinations(
        include_derived=include_derived_features,
    )
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
    progress.add_log_message(
        f"[{datetime.now().strftime('%H:%M:%S')}] Optimizer started"
    )
    progress.add_log_message(
        f"[{datetime.now().strftime('%H:%M:%S')}] Original settings saved"
    )
    progress.add_log_message(
        f"[{datetime.now().strftime('%H:%M:%S')}] Testing {len(combinations)} configurations with 2 models each ({total_configs} total trainings)"
    )
    
    # Log memory settings
    mem_limit_str = f"{max_memory_mb:.0f} MB" if max_memory_mb else "1536 MB (default)"
    progress.add_log_message(
        f"[{datetime.now().strftime('%H:%M:%S')}] Memory limit: {mem_limit_str}, Max workers: {max_workers}"
    )
    progress.add_log_message(
        f"[{datetime.now().strftime('%H:%M:%S')}] Using adaptive parallelism with memory-based throttling"
    )
    
    if progress_callback:
        progress_callback(progress)
    
    try:
        progress.phase = "training"
        
        # Log initial memory state
        _log_memory_usage("Optimizer start")
        
        # Create a list of all training tasks (config + model_type combinations)
        training_tasks = []
        for combo in combinations:
            config_name = _configuration_to_name(combo)
            training_tasks.append((config_name, combo, "single_step", train_single_step_fn))
            training_tasks.append((config_name, combo, "two_step", train_two_step_fn))
        
        # ADAPTIVE PARALLEL PROCESSING with memory throttling
        # Uses ThreadPoolExecutor but limits active workers based on memory usage
        # Falls back to sequential when memory is constrained
        _Logger.info(
            "Processing %d training tasks with adaptive parallelism (max %d workers, memory limit %s)",
            len(training_tasks), max_workers, mem_limit_str
        )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Track active futures and pending tasks
            active_futures = {}
            pending_tasks = list(enumerate(training_tasks, start=1))
            pending_tasks.reverse()  # Use as a stack (pop from end)
            
            # Process tasks with memory-aware throttling
            while pending_tasks or active_futures:
                # Submit new tasks if memory allows and we have capacity
                while (
                    len(active_futures) < max_workers 
                    and pending_tasks 
                    and _should_allow_parallel_task(max_memory_mb)
                ):
                    idx, (config_name, combo, model_type, train_fn) = pending_tasks.pop()
                    
                    future = executor.submit(
                        _train_single_configuration,
                        config_name,
                        combo,
                        model_type,
                        train_fn,
                        build_dataset_fn,
                        min_samples,
                    )
                    active_futures[future] = (idx, config_name, model_type)
                    _Logger.debug("Submitted task %d/%d: %s (%s)", idx, total_configs, config_name, model_type)
                
                # Wait for at least one task to complete if we have any active
                if active_futures:
                    # Use timeout to periodically check memory even if no tasks complete
                    done, _ = concurrent.futures.wait(
                        active_futures.keys(),
                        timeout=2.0,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    # Process completed tasks
                    for future in done:
                        idx, config_name, model_type = active_futures.pop(future)
                        
                        try:
                            result = future.result()
                            
                            # Log memory periodically
                            if idx % 10 == 1:
                                _log_memory_usage(f"After task {idx}/{total_configs}")
                            
                            # Update progress
                            with _progress_lock:
                                progress.results.append(result)
                                progress.completed_configurations += 1
                                progress.current_configuration = config_name
                                progress.current_model_type = model_type
                                
                                # Log the result
                                if result.success:
                                    mape_str = f"{result.val_mape_pct:.2f}%" if result.val_mape_pct is not None else "N/A"
                                    progress.add_log_message(
                                        f"[{datetime.now().strftime('%H:%M:%S')}] [{progress.completed_configurations}/{total_configs}] {config_name} ({model_type}): Val MAPE = {mape_str}"
                                    )
                                else:
                                    progress.add_log_message(
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
                                        progress.add_log_message(
                                            f"[{datetime.now().strftime('%H:%M:%S')}] 🏆 New best: {config_name} ({model_type}) with Val MAPE = {result.val_mape_pct:.2f}%"
                                        )
                            
                            # Call progress callback outside the lock to avoid deadlocks
                            if progress_callback:
                                progress_callback(progress)
                            
                            # Garbage collection every 10 completed tasks
                            if progress.completed_configurations % 10 == 0:
                                gc.collect()
                                _log_memory_usage(f"After GC at {progress.completed_configurations}/{total_configs}")
                                _Logger.info("Garbage collection completed at %d/%d", progress.completed_configurations, total_configs)
                                
                        except Exception as e:
                            _Logger.error("Error processing result for %s (%s): %s", config_name, model_type, e)
                            with _progress_lock:
                                progress.add_log_message(
                                    f"[{datetime.now().strftime('%H:%M:%S')}] Error processing result: {str(e)}"
                                )
                
                # If memory is high and no tasks completed, wait a bit before checking again
                elif pending_tasks:
                    _Logger.debug("Memory limit reached, waiting for tasks to complete...")
                    time.sleep(1.0)
        
        # Final memory log
        _log_memory_usage("Optimizer complete")
        
        # Optimization complete
        progress.phase = "complete"
        progress.end_time = datetime.now()
        
        with _progress_lock:
            if progress.best_result:
                progress.add_log_message(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Optimization complete!"
                )
                progress.add_log_message(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Best configuration: {progress.best_result.config_name}"
                )
                progress.add_log_message(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Best model: {progress.best_result.model_type}"
                )
                mape_str = f"{progress.best_result.val_mape_pct:.2f}%" if progress.best_result.val_mape_pct else "N/A"
                progress.add_log_message(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Best Val MAPE: {mape_str}"
                )
            else:
                progress.add_log_message(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Optimization complete (no valid results)"
                )
        
    except Exception as e:
        _Logger.error("Optimizer error: %s", e, exc_info=True)
        progress.phase = "error"
        progress.error_message = str(e)
        with _progress_lock:
            progress.add_log_message(
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
                progress.add_log_message(
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
    
    This function handles both experimental features and derived features
    using the generic enable_feature()/disable_feature() API from FeatureConfiguration.
    
    Note: This relies on FeatureConfiguration.enable_feature() and disable_feature()
    methods which support both experimental and derived features (see ml/feature_config.py).
    
    Args:
        best_result: The best OptimizationResult from optimization
        enable_two_step: If True, also enable two-step prediction mode
        
    Returns:
        True if successfully applied and saved
    """
    try:
        config = get_feature_config()
        
        # Apply feature settings using generic enable_feature()/disable_feature()
        # These methods work for both experimental and derived features
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
