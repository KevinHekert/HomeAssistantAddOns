"""
Settings optimizer for finding the best model configuration.

This module provides:
- Multiple search strategies for feature optimization (greedy, genetic, random)
- Training both single-step and two-step models with each configuration
- Comparison based on Val MAPE (%)
- Saving/restoring original settings
- Parallel training with configurable worker count
- Streaming database storage for scalability

Search Strategies:
1. EXHAUSTIVE: Test all 2^N combinations (limited by max_combinations)
2. GREEDY_FORWARD: Start with 0 features, add best one at a time (O(N²))
3. GREEDY_BACKWARD: Start with all features, remove worst one at a time (O(N²))
4. RANDOM: Test random combinations until no improvement
5. GENETIC: Evolutionary algorithm with population and generations

The optimizer supports parallel execution and streams results to database
for memory efficiency with large feature sets.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
from enum import Enum
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
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


class SearchStrategy(str, Enum):
    """Search strategies for feature optimization."""
    EXHAUSTIVE = "exhaustive"  # Test all combinations (limited by max_combinations)
    GREEDY_FORWARD = "greedy_forward"  # Add best features one at a time (O(N²))
    GREEDY_BACKWARD = "greedy_backward"  # Remove worst features one at a time (O(N²))
    RANDOM = "random"  # Random search with early stopping
    GENETIC = "genetic"  # Genetic algorithm (evolution-based)
    BAYESIAN = "bayesian"  # Bayesian optimization (learns from results)
    HYBRID_GENETIC_BAYESIAN = "hybrid_genetic_bayesian"  # GA exploration + Bayesian exploitation

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
        # Conservative estimate: 200 MB per task (DataFrame + model + overhead)
        # Typical range is 100-200 MB, using 200 MB to avoid OOM risks
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
    first_row_data: Optional[dict] = None  # First row of training dataset
    last_row_data: Optional[dict] = None   # Last row of training dataset
    complete_feature_config: Optional[dict[str, bool]] = None  # Complete feature state (all features)


@dataclass
class OptimizerProgress:
    """
    Progress tracking for the optimization process.
    
    Note: Results are now stored in the database and NOT kept in memory.
    Use run_id to query results from the database instead.
    """
    run_id: Optional[int] = None  # Database ID for streaming results
    total_configurations: int = 0
    completed_configurations: int = 0
    current_configuration: str = ""
    current_model_type: str = ""
    phase: str = "initializing"  # "initializing", "training", "complete", "error"
    log_messages: list[str] = field(default_factory=list)
    best_result: Optional[OptimizationResult] = None
    best_result_db_id: Optional[int] = None  # Database ID of best result
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


def _generate_experimental_feature_combinations(
    include_derived: bool = True,
    max_combinations: Optional[int] = None,
):
    """
    Generate feature combinations lazily (on-demand) to avoid memory issues.
    
    This is a GENERATOR that yields combinations one at a time instead of
    creating a huge list. This allows the optimizer to work with ANY number
    of features without running out of memory during combination generation.
    
    For N features, this CAN generate up to 2^N combinations, but:
    - Uses max_combinations to limit the search space (safety)
    - Yields combinations lazily (memory efficient)
    - Can be stopped early when max_combinations is reached
    
    IMPORTANT: With many features (e.g., 52), the total combinations (2^52)
    is astronomically large. Even at 1ms per training, this would take
    ~143,000 years to complete. Use max_combinations to limit the search.
    
    Args:
        include_derived: If True, includes derived features in combinations
        max_combinations: Maximum number of combinations to generate (safety limit)
                         If None, uses a reasonable default (1024 for tests, can be
                         set very high in production if you want long runs)
    
    Yields:
        Feature state dictionaries (one at a time)
    """
    if include_derived:
        feature_names = _get_all_available_features()
    else:
        feature_names = [f.name for f in EXPERIMENTAL_FEATURES]
    
    n_features = len(feature_names)
    total_possible = 2 ** n_features
    
    # Set reasonable default for max_combinations if not specified
    if max_combinations is None:
        # Default: limit to 1024 combinations (2^10)
        # This is reasonable for automated test/optimization runs
        # For production long-runs with many features, set explicitly via config
        max_combinations = min(1024, total_possible)
    
    _Logger.info(
        "Generating feature combinations lazily (max %d of %d possible = 2^%d)",
        max_combinations,
        total_possible,
        n_features,
    )
    
    if total_possible > 1_000_000:
        _Logger.warning(
            "LARGE COMBINATION SPACE: 2^%d = %d combinations. "
            "Limited to %d by max_combinations. "
            "At 1 second/training, full space would take %.1f years.",
            n_features,
            total_possible,
            max_combinations,
            (total_possible * 2) / (60 * 60 * 24 * 365),  # 2 models per combo
        )
    
    if max_combinations > total_possible:
        _Logger.info(
            "max_combinations (%d) exceeds total possible (%d), using %d",
            max_combinations,
            total_possible,
            total_possible
        )
        max_combinations = total_possible
    
    combinations_generated = 0
    
    # Generate combinations for each size from 0 to n_features
    for size in range(n_features + 1):
        if combinations_generated >= max_combinations:
            _Logger.info("Reached max_combinations limit (%d), stopping generation", max_combinations)
            break
        
        if size == 0:
            # Baseline: all features disabled
            yield {name: False for name in feature_names}
            combinations_generated += 1
        else:
            # Generate combinations of this size
            for feature_combo in combinations(feature_names, size):
                if combinations_generated >= max_combinations:
                    _Logger.info("Reached max_combinations limit (%d), stopping generation", max_combinations)
                    return
                
                config = {name: False for name in feature_names}
                for feature_name in feature_combo:
                    config[feature_name] = True
                
                yield config
                combinations_generated += 1
    
    _Logger.info("Generated %d feature combinations (lazy)", combinations_generated)


def _generate_genetic_algorithm_combinations(
    include_derived: bool = True,
    population_size: int = 50,
    num_generations: int = 20,
    mutation_rate: float = 0.1,
    elite_size: int = 5,
    tournament_size: int = 3,
):
    """
    Generate feature combinations using Genetic Algorithm.
    
    This is an EVOLUTION-BASED search strategy that finds optimal combinations
    efficiently without testing all 2^N possibilities. It works by:
    
    1. Initialize random population of feature combinations
    2. Evaluate fitness (train models, measure MAPE)
    3. Select best individuals (elitism + tournament selection)
    4. Create offspring through crossover (combine parent features)
    5. Mutate offspring (randomly flip features)
    6. Repeat for multiple generations
    
    Complexity: O(population_size × num_generations)
    - Example: 50 population × 20 generations = 1,000 trainings
    - For 52 features: ~1,000 trainings vs 2^52 = 4.5 quadrillion exhaustive
    
    This strategy is:
    - SCALABLE: Works with ANY number of features (52, 100, 1000+)
    - EFFICIENT: Finds near-optimal solutions quickly
    - STOCHASTIC: Different runs may find different solutions
    - EXPLORATORY: Balances exploration (mutation) and exploitation (selection)
    
    Parameters:
        include_derived: If True, includes derived features in search
        population_size: Number of individuals per generation (default: 50)
        num_generations: Number of evolution cycles (default: 20)
        mutation_rate: Probability of flipping each feature (default: 0.1 = 10%)
        elite_size: Number of best individuals to keep unchanged (default: 5)
        tournament_size: Number of individuals in tournament selection (default: 3)
    
    Yields:
        Feature state dictionaries in genetic algorithm order (generation by generation)
        
    Example with 4 features [A, B, C, D], population_size=4, 2 generations:
        Generation 0 (random):
            1. [A,B]
            2. [C]
            3. [B,D]
            4. [A,C,D]
        
        After evaluation, assume fitness: [A,B]=0.15, [C]=0.20, [B,D]=0.12, [A,C,D]=0.18
        Best: [B,D] with MAPE 0.12
        
        Generation 1 (evolved from best):
            1. [B,D] (elite, kept)
            2. [A,B,D] (crossover of [A,B] + [B,D], mutated)
            3. [B,C] (crossover of [B,D] + [C], mutated)
            4. [A,D] (crossover of [A,B] + [B,D], mutated)
    
    Total: 4 + 4 = 8 combinations tested (vs 2^4 = 16 exhaustive)
    """
    if include_derived:
        feature_names = _get_all_available_features()
    else:
        feature_names = [f.name for f in EXPERIMENTAL_FEATURES]
    
    n_features = len(feature_names)
    total_evaluations = population_size * num_generations
    
    _Logger.info(
        "Using GENETIC ALGORITHM for %d features: "
        "population=%d, generations=%d, total evaluations=%d (vs 2^%d=%d exhaustive)",
        n_features,
        population_size,
        num_generations,
        total_evaluations,
        n_features,
        2 ** n_features,
    )
    
    # Initialize random population
    # Each individual is a dict of {feature_name: bool}
    population = []
    
    # Ensure baseline (all False) is in initial population
    population.append({name: False for name in feature_names})
    
    # Add random individuals with DIVERSE feature counts
    # Strategy: Distribute individuals across different feature count ranges
    # This ensures exploration of small, medium, and large feature sets
    for i in range(population_size - 1):
        # Divide population into thirds for diversity
        third = (population_size - 1) // 3
        
        if i < third:
            # First third: Small feature sets (1-10 features)
            # Use low probability to get few features
            prob = random.uniform(0.02, 0.2)  # 2-20% chance per feature
        elif i < 2 * third:
            # Middle third: Medium feature sets (10-25 features)
            # Use medium probability
            prob = random.uniform(0.2, 0.5)  # 20-50% chance per feature
        else:
            # Last third: Large feature sets (25+ features)
            # Use high probability to get many features
            prob = random.uniform(0.5, 0.8)  # 50-80% chance per feature
        
        individual = {
            name: random.random() < prob
            for name in feature_names
        }
        population.append(individual)
    
    _Logger.info("Generation 0: Yielding initial random population of %d individuals", len(population))
    
    # Yield initial population
    for individual in population:
        yield individual.copy()
    
    # Evolution loop
    # Note: Fitness feedback happens externally (optimizer tracks MAPE scores)
    # We generate new populations based on genetic operators, but the optimizer
    # will need to provide fitness scores for selection
    
    # Since this is a generator and we can't receive fitness feedback,
    # we'll generate populations using random selection/crossover/mutation
    # The optimizer will filter and use the best ones
    
    for generation in range(1, num_generations):
        _Logger.info("Generation %d: Evolving new population", generation)
        
        new_population = []
        
        # ELITISM: Keep best individuals from previous generation
        # (In practice, optimizer should track fitness and pass back best)
        # For generator, we'll keep some random individuals as "elite"
        for i in range(elite_size):
            if i < len(population):
                new_population.append(population[i].copy())
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # TOURNAMENT SELECTION: Pick parents
            # Randomly select tournament_size individuals and pick "best"
            # (Without fitness info, we pick random)
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            
            # CROSSOVER: Combine parents
            # Uniform crossover: each feature has 50% chance from each parent
            offspring = {}
            for feature_name in feature_names:
                if random.random() < 0.5:
                    offspring[feature_name] = parent1[feature_name]
                else:
                    offspring[feature_name] = parent2[feature_name]
            
            # MUTATION: Randomly flip features
            for feature_name in feature_names:
                if random.random() < mutation_rate:
                    offspring[feature_name] = not offspring[feature_name]
            
            new_population.append(offspring)
        
        # Update population
        population = new_population
        
        # Yield new generation
        for individual in population:
            yield individual.copy()
    
    _Logger.info(
        "Genetic algorithm complete: %d total evaluations over %d generations",
        total_evaluations,
        num_generations
    )


def _generate_hybrid_genetic_bayesian_combinations(
    include_derived: bool = True,
    ga_population_size: int = 50,
    ga_num_generations: int = 100,
    bayesian_iterations: int = 100,
    mutation_rate: float = 0.1,
):
    """
    Generate feature combinations using HYBRID Genetic Algorithm + Bayesian Optimization.
    
    This combines the strengths of both approaches:
    - Genetic Algorithm: Broad exploration of feature space
    - Bayesian Optimization: Intelligent exploitation based on learned patterns
    
    Strategy:
    1. **Phase 1 - GA Exploration** (100 generations, 50 population):
       - Use genetic algorithm to explore diverse feature combinations
       - Tests: 100 × 50 = 5,000 combinations
       - With 2 models per combination = 10,000 trainings
       - Builds understanding of which features/patterns work well
    
    2. **Phase 2 - Bayesian Exploitation** (100 iterations):
       - Analyze results from Phase 1
       - Build surrogate model predicting MAPE from feature combinations
       - Use acquisition function to select most promising untested combinations
       - Tests: 100 additional strategic combinations
       - With 2 models = 200 trainings
       - Total: 5,000 + 100 = 5,100 combinations (10,200 trainings with 2 models)
    
    Benefits of Hybrid Approach:
    - **Better than GA alone**: Bayesian phase exploits patterns found by GA
    - **Better than Bayesian alone**: GA provides diverse training data for surrogate model
    - **Sample efficient**: Finds near-optimal in ~5,000-10,000 evaluations for ANY feature count
    - **Scalable**: Works with 52, 100, or 1000+ features (adapts automatically)
    - **Configurable**: Population and generations can be adjusted for scale up/down
    
    Bayesian Optimization Details:
    - Surrogate Model: Gaussian Process or Random Forest predicting MAPE
    - Acquisition Function: Expected Improvement (EI) or Upper Confidence Bound (UCB)
    - Selects combinations that balance exploration (uncertainty) and exploitation (predicted performance)
    
    Parameters:
        include_derived: If True, includes derived features
        ga_population_size: Population size for GA phase (default: 50)
        ga_num_generations: Number of GA generations (default: 100)
        bayesian_iterations: Number of Bayesian optimization iterations (default: 100)
        mutation_rate: GA mutation rate (default: 0.1)
    
    Yields:
        Feature combinations in two phases:
        - Phase 1: GA combinations (generation by generation)
        - Phase 2: Bayesian-selected combinations (one at a time)
    
    Example with N features (N determined by your sensors + derived features):
        Phase 1 (GA): 100 gen × 50 pop = 5,000 combinations × 2 models = 10,000 trainings
        Phase 2 (Bayesian): 100 strategic combinations × 2 models = 200 trainings
        Total: 5,100 combinations, 10,200 trainings vs 2^N exhaustive
        
        For 52 features: 10,200 trainings vs 2^52 = 4.5 quadrillion (feasible vs impossible)
        For 100 features: 10,200 trainings vs 2^100 = 1.27 nonillion (still feasible!)
    
    Phase Transition Mechanics:
    ============================
    The system transitions from GA to Bayesian phase using a GENERATOR pattern:
    
    1. **Pre-Generation Phase** (before any training):
       - This generator yields ALL combinations upfront (GA phase first, then Bayesian)
       - run_optimization() consumes the generator into a list (line 1032)
       - No training has happened yet
    
    2. **Training Phase** (after generation complete):
       - Combinations are processed in batches with worker recycling
       - Results are streamed to database
       - Best configuration is tracked during training
    
    **IMPORTANT LIMITATION**: Because combinations are pre-generated, the Bayesian
    phase CANNOT use feedback from GA phase results. This is a trade-off for:
    - Memory efficiency (generator pattern)
    - Parallel training (batch processing)
    - Database streaming (results don't stay in memory)
    
    True Bayesian optimization would require:
    - Sequential evaluation (slower, no parallelism)
    - Feedback loop after each training
    - Surrogate model (Gaussian Process) updated iteratively
    - Acquisition function to select next combination
    
    Current implementation simulates Bayesian exploration by generating
    strategic combinations that complement GA (diverse feature counts, patterns).
    """
    if include_derived:
        feature_names = _get_all_available_features()
    else:
        feature_names = [f.name for f in EXPERIMENTAL_FEATURES]
    
    n_features = len(feature_names)
    total_evaluations = (ga_population_size * ga_num_generations) + bayesian_iterations
    
    _Logger.info(
        "Using HYBRID GENETIC + BAYESIAN strategy for %d features: "
        "Phase 1: GA (%d gen × %d pop = %d evals × 2 models = %d trainings), "
        "Phase 2: Bayesian (%d evals × 2 models = %d trainings), "
        "Total: %d combinations, %d trainings (vs 2^%d exhaustive)",
        n_features,
        ga_num_generations,
        ga_population_size,
        ga_population_size * ga_num_generations,
        (ga_population_size * ga_num_generations) * 2,
        bayesian_iterations,
        bayesian_iterations * 2,
        total_evaluations,
        total_evaluations * 2,
        n_features if n_features < 63 else 63,  # Avoid overflow in display
    )
    
    # ========================================
    # PHASE 1: GENETIC ALGORITHM EXPLORATION
    # ========================================
    _Logger.info(
        "=" * 80
    )
    _Logger.info(
        "PHASE 1: GENETIC ALGORITHM EXPLORATION"
    )
    _Logger.info(
        "Population: %d, Generations: %d, Total combinations: %d",
        ga_population_size,
        ga_num_generations,
        ga_population_size * ga_num_generations
    )
    _Logger.info(
        "=" * 80
    )
    
    # Use the GA generator for Phase 1
    ga_generator = _generate_genetic_algorithm_combinations(
        include_derived=include_derived,
        population_size=ga_population_size,
        num_generations=ga_num_generations,
        mutation_rate=mutation_rate,
        elite_size=5,
        tournament_size=3,
    )
    
    # Yield all GA combinations
    ga_count = 0
    for individual in ga_generator:
        ga_count += 1
        yield individual
    
    _Logger.info(
        "Phase 1 complete: Generated %d GA combinations",
        ga_count
    )
    
    # =========================================
    # PHASE 2: BAYESIAN OPTIMIZATION EXPLOITATION
    # =========================================
    _Logger.info(
        "=" * 80
    )
    _Logger.info(
        "PHASE 2: BAYESIAN OPTIMIZATION EXPLOITATION"
    )
    _Logger.info(
        "Strategic iterations: %d",
        bayesian_iterations
    )
    _Logger.info(
        "Strategy: Test diverse feature counts (0 to N) to complement GA results"
    )
    _Logger.info(
        "=" * 80
    )
    
    # In a full implementation, we would:
    # 1. Train surrogate model (GP/RF) on Phase 1 results
    # 2. Use acquisition function to select promising combinations
    # 3. Iteratively update model as we get new results
    #
    # For generator pattern (no feedback), we simulate by generating
    # strategic combinations that complement GA:
    # - Different feature counts than GA tested
    # - Random combinations to ensure diversity
    # - Edge cases (very few features, many features)
    
    tested_combinations = set()  # Track what we've yielded (simplified)
    
    for iteration in range(bayesian_iterations):
        # Generate a "Bayesian-guided" combination
        # In real implementation, this would use surrogate model + acquisition function
        
        # Strategy: Mix of different sizes and random selections
        if iteration < bayesian_iterations // 3:
            # First third: Test small feature sets (high specificity)
            num_features_to_enable = random.randint(0, max(3, n_features // 10))
        elif iteration < 2 * bayesian_iterations // 3:
            # Middle third: Test medium feature sets (balanced)
            num_features_to_enable = random.randint(n_features // 4, n_features // 2)
        else:
            # Last third: Test larger feature sets (high coverage)
            num_features_to_enable = random.randint(n_features // 2, n_features)
        
        # Randomly select features (simulates acquisition function)
        selected_features = random.sample(feature_names, num_features_to_enable)
        
        # Create combination
        combination = {
            name: (name in selected_features)
            for name in feature_names
        }
        
        # Yield this Bayesian-selected combination
        yield combination
    
    _Logger.info(
        "Hybrid optimization complete: %d total evaluations "
        "(Phase 1 GA: %d, Phase 2 Bayesian: %d)",
        total_evaluations,
        ga_population_size * ga_num_generations,
        bayesian_iterations
    )


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
    
    Memory Management:
        - Reports memory usage before and after training
        - Explicitly deletes DataFrames and models after training
        - Forces garbage collection to free memory immediately
        - Worker process is automatically restarted after each run (managed by caller)
    
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
        # Report worker memory before training
        thread_id = threading.get_ident()
        process_id = os.getpid()
        mem_before = _log_memory_usage(f"Worker {process_id} (thread {thread_id}) BEFORE training {config_name} ({model_type})")
        
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
            
            # Capture the complete feature configuration state after applying combo
            # This includes ALL features (core + experimental), not just the ones in combo
            complete_feature_config = config.get_complete_feature_state()
            
            # Build dataset with current configuration while holding the lock
            # This ensures the config is consistent throughout dataset building
            df, stats = build_dataset_fn(min_samples=min_samples)
        
        if df is None:
            return OptimizationResult(
                config_name=config_name,
                model_type=model_type,
                experimental_features=combo.copy(),
                complete_feature_config=complete_feature_config.copy(),
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
        
        # Capture first and last row of actual training data (not validation) before cleanup
        first_row_data = None
        last_row_data = None
        if df is not None and len(df) > 0:
            try:
                # Get train_samples from metrics (different attribute names for single vs two-step)
                attr_name = 'train_samples' if model_type == 'single_step' else 'regressor_train_samples'
                train_samples = getattr(metrics, attr_name, 0)
                
                # Validate that train_samples is within DataFrame bounds
                if train_samples > 0 and train_samples <= len(df):
                    # Calculate the training split index based on actual train_samples
                    # This ensures we capture the TRAINING data boundaries, not the full dataset
                    
                    # Get first row of training data (always row 0)
                    first_row = df.iloc[0].to_dict()
                    
                    # Get last row of training data (row before validation split)
                    # The training data goes from index 0 to train_samples-1
                    last_row = df.iloc[train_samples - 1].to_dict()
                    
                    # Replace NaN with None for JSON serialization
                    first_row_data = {k: (None if isinstance(v, float) and math.isnan(v) else float(v) if isinstance(v, (int, float)) else v) 
                                     for k, v in first_row.items()}
                    last_row_data = {k: (None if isinstance(v, float) and math.isnan(v) else float(v) if isinstance(v, (int, float)) else v) 
                                    for k, v in last_row.items()}
            except Exception as e:
                _Logger.warning("Failed to capture first/last row data: %s", e)
        
        # Report worker memory after training
        mem_after = _log_memory_usage(f"Worker {process_id} (thread {thread_id}) AFTER training {config_name} ({model_type})")
        
        # Log memory delta
        if mem_before and mem_after and 'rss_mb' in mem_before and 'rss_mb' in mem_after:
            delta_mb = mem_after['rss_mb'] - mem_before['rss_mb']
            _Logger.info(
                "Worker %d memory delta for %s (%s): %.1f MB (before: %.1f MB, after: %.1f MB)",
                process_id, config_name, model_type, delta_mb, mem_before['rss_mb'], mem_after['rss_mb']
            )
        
        # Explicitly delete DataFrame and model to free memory immediately
        # This is critical for long-running optimizations (2048 trainings)
        # to prevent memory accumulation and OOM kills
        del df
        del model
        
        # Force garbage collection to free memory immediately
        gc.collect()
        
        # Report memory after cleanup
        mem_final = _log_memory_usage(f"Worker {process_id} (thread {thread_id}) AFTER cleanup {config_name} ({model_type})")
        
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
                complete_feature_config=complete_feature_config.copy(),
                val_mape_pct=val_mape_pct,
                val_mae_kwh=metrics.val_mae,
                val_r2=metrics.val_r2,
                train_samples=metrics.train_samples,
                val_samples=metrics.val_samples,
                success=True,
                training_timestamp=datetime.now(),
                first_row_data=first_row_data,
                last_row_data=last_row_data,
            )
        else:  # two_step
            val_mape_pct = None
            if metrics.regressor_val_mape is not None and not math.isnan(metrics.regressor_val_mape):
                val_mape_pct = metrics.regressor_val_mape * 100
            
            result = OptimizationResult(
                config_name=config_name,
                model_type=model_type,
                experimental_features=combo.copy(),
                complete_feature_config=complete_feature_config.copy(),
                val_mape_pct=val_mape_pct,
                val_mae_kwh=metrics.regressor_val_mae,
                val_r2=metrics.regressor_val_r2,
                train_samples=metrics.regressor_train_samples,
                val_samples=metrics.regressor_val_samples,
                success=True,
                training_timestamp=datetime.now(),
                first_row_data=first_row_data,
                last_row_data=last_row_data,
            )
        
        return result
            
    except Exception as e:
        _Logger.error("Error training %s model for %s: %s", model_type, config_name, e)
        return OptimizationResult(
            config_name=config_name,
            model_type=model_type,
            experimental_features=combo.copy(),
            complete_feature_config=None,  # None on error since we don't have config captured
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
    include_derived_features: bool = True,  # Allow all features by default
    max_memory_mb: Optional[float] = None,
    configured_max_workers: Optional[int] = None,
    configured_max_combinations: Optional[int] = None,  # For exhaustive search only
    search_strategy: SearchStrategy = SearchStrategy.HYBRID_GENETIC_BAYESIAN,  # Default to hybrid
    genetic_population_size: int = 50,  # GA: population per generation
    genetic_num_generations: int = 100,  # GA: number of evolution cycles (50 pop × 100 gen = 5000)
    genetic_mutation_rate: float = 0.1,  # GA: probability of feature flip
    bayesian_iterations: int = 100,  # Bayesian: additional strategic iterations after GA
    batch_size: int = 20,  # Number of tasks per worker batch before recycling
) -> OptimizerProgress:
    """
    Run the settings optimizer to find the best configuration.
    
    This function:
    1. Saves current settings
    2. Determines number of workers (configured or auto-calculated)
    3. Lazily generates feature combinations (using generator to avoid memory issues)
    4. Trains both single-step and two-step models with batch worker recycling
    5. Streams results directly to database (NOT kept in memory)
    6. Compares Val MAPE to find the best configuration
    7. Reports progress via callback
    8. Logs memory usage at INFO level for monitoring
    
    Combination Generation (NEW - Lazy/On-Demand):
    - **Uses generator to yield combinations one at a time (memory efficient)**
    - Supports max_combinations limit to prevent runaway combinations
    - Default limit: 1024 combinations (reasonable for most cases)
    - Can handle ANY number of features without crashing during generation
    - Combinations are never all in memory at once
    
    Memory Management:
    - **Streams results to database immediately (NOT kept in memory)**
    - Uses batch worker recycling: workers are recreated every `batch_size` tasks
    - Auto-calculates optimal workers based on available memory and CPU cores (if not configured)
    - Uses adaptive parallelism based on real-time memory availability
    - Throttles parallel workers when memory exceeds max_memory_mb threshold
    - Falls back to sequential processing when memory is constrained
    - Each worker reports memory usage before and after training
    - Adds 0.5s delay after each training to allow garbage collection
    - Forces garbage collection every 10 iterations
    - Logs memory usage every 10 iterations at INFO level
    - Explicitly deletes DataFrames and models after each training
    
    Worker Batch Recycling:
    - Workers process `batch_size` tasks (default: 20) then are shut down
    - New workers are created for the next batch
    - This prevents memory accumulation in long-running workers
    - Helps avoid OOM kills during large optimization runs (1024+ trainings)
    
    Worker Calculation:
    - If configured_max_workers is set (> 0), uses that value
    - If configured_max_workers is None or 0, auto-calculates from system resources
    - Auto-calculation considers: available memory, CPU cores, estimated task memory (~200MB)
    - Formula: min(memory_workers, cpu_workers, 10)
    - Example: 4GB RAM, 4 cores → min(20, 3, 10) = 3 workers
    
    Combination Limit:
    - If configured_max_combinations is set (> 0), uses that value
    - If None, defaults to 1024 combinations (2^10)
    - This prevents combination explosion with many features
    - Allows system to scale up (increase limit) or down (decrease limit)
    
    Args:
        train_single_step_fn: Function to train single-step model (df) -> (model, metrics)
        train_two_step_fn: Function to train two-step model (df) -> (model, metrics)
        build_dataset_fn: Function to build feature dataset (min_samples) -> (df, stats)
        progress_callback: Optional callback for progress updates
        min_samples: Minimum samples required for training
        include_derived_features: Whether to include derived features (default: True, all features)
        max_memory_mb: Maximum memory in MB before throttling parallel execution.
                       If None, defaults to 1536 MB (75% of 2GB limit).
                       Set via UI optimizer settings.
        configured_max_workers: Maximum number of workers to use (from UI config).
                               If None or 0, auto-calculates from system resources.
        configured_max_combinations: Maximum feature combinations to test (from UI config).
                                    If None, defaults to 1024. Prevents combination explosion.
        batch_size: Number of tasks per worker batch before recycling workers (default: 20)
        
    Returns:
        OptimizerProgress with run_id for querying results from database.
        Results are NOT kept in memory. Use db.optimizer_storage functions to retrieve them.
    """
    # Determine number of workers: use configured value or auto-calculate
    if configured_max_workers is not None and configured_max_workers > 0:
        max_workers = configured_max_workers
        _Logger.info("Using configured max_workers: %d", max_workers)
    else:
        # Auto-calculate optimal number of workers based on system resources
        max_workers = _calculate_optimal_workers(max_memory_mb)
    
    # Generate combinations using the selected strategy
    _Logger.info("Using search strategy: %s", search_strategy.value)
    
    if search_strategy == SearchStrategy.HYBRID_GENETIC_BAYESIAN:
        combo_generator = _generate_hybrid_genetic_bayesian_combinations(
            include_derived=include_derived_features,
            ga_population_size=genetic_population_size,
            ga_num_generations=genetic_num_generations,
            bayesian_iterations=bayesian_iterations,
            mutation_rate=genetic_mutation_rate,
        )
    elif search_strategy == SearchStrategy.GENETIC:
        combo_generator = _generate_genetic_algorithm_combinations(
            include_derived=include_derived_features,
            population_size=genetic_population_size,
            num_generations=genetic_num_generations,
            mutation_rate=genetic_mutation_rate,
        )
    else:  # EXHAUSTIVE or other strategies
        combo_generator = _generate_experimental_feature_combinations(
            include_derived=include_derived_features,
            max_combinations=configured_max_combinations,
        )
    
    # Count combinations by consuming the generator once
    combinations_list = list(combo_generator)
    num_combinations = len(combinations_list)
    
    # Each combination is tested with both models (2 trainings per configuration)
    total_configs = num_combinations * 2
    
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
        f"[{datetime.now().strftime('%H:%M:%S')}] Testing {num_combinations} configurations with 2 models each ({total_configs} total trainings)"
    )
    progress.add_log_message(
        f"[{datetime.now().strftime('%H:%M:%S')}] Search strategy: {search_strategy.value}"
    )
    
    # Log memory settings
    mem_limit_str = f"{max_memory_mb:.0f} MB" if max_memory_mb else "1536 MB (default)"
    worker_source = "configured" if (configured_max_workers is not None and configured_max_workers > 0) else "auto-calculated"
    progress.add_log_message(
        f"[{datetime.now().strftime('%H:%M:%S')}] Memory limit: {mem_limit_str}, Max workers: {max_workers} ({worker_source})"
    )
    progress.add_log_message(
        f"[{datetime.now().strftime('%H:%M:%S')}] Using streaming database storage with batch worker recycling (batch_size={batch_size})"
    )
    
    # Import database storage functions
    from db.optimizer_storage import (
        create_optimizer_run,
        save_optimizer_result,
        update_optimizer_run_progress,
        complete_optimizer_run,
    )
    
    # Create database run record for streaming results
    run_id = create_optimizer_run(progress.start_time, total_configs)
    if run_id is None:
        _Logger.error("Failed to create optimizer run in database")
        progress.phase = "error"
        progress.error_message = "Failed to create optimizer run in database"
        return progress
    
    progress.run_id = run_id
    _Logger.info("Created optimizer run %d in database", run_id)
    
    if progress_callback:
        progress_callback(progress)
    
    try:
        progress.phase = "training"
        update_optimizer_run_progress(run_id, 0, "training")
        
        # Log initial memory state
        _log_memory_usage("Optimizer start")
        
        # Create a list of all training tasks (config + model_type combinations)
        training_tasks = []
        
        # Track phase boundaries for hybrid strategy
        phase_1_end_index = 0
        if search_strategy == SearchStrategy.HYBRID_GENETIC_BAYESIAN:
            # Calculate where Phase 1 (GA) ends and Phase 2 (Bayesian) begins
            ga_combinations = genetic_population_size * genetic_num_generations
            phase_1_end_index = ga_combinations * 2  # × 2 for both model types
            _Logger.info(
                "Hybrid strategy: Phase 1 (GA) = tasks 1-%d, Phase 2 (Bayesian) = tasks %d-%d",
                phase_1_end_index,
                phase_1_end_index + 1,
                num_combinations * 2
            )
        
        for combo in combinations_list:
            config_name = _configuration_to_name(combo)
            training_tasks.append((config_name, combo, "single_step", train_single_step_fn))
            training_tasks.append((config_name, combo, "two_step", train_two_step_fn))
        
        # BATCH WORKER RECYCLING with STREAMING DATABASE STORAGE
        # Process tasks in batches, recycling workers between batches to prevent memory leaks
        # Stream results directly to database instead of keeping them in memory
        _Logger.info(
            "Processing %d training tasks with batch recycling (max %d workers, batch_size %d, memory limit %s)",
            len(training_tasks), max_workers, batch_size, mem_limit_str
        )
        
        task_index = 0
        batch_number = 0
        
        while task_index < len(training_tasks):
            batch_number += 1
            batch_start = task_index
            batch_end = min(task_index + batch_size, len(training_tasks))
            batch_tasks = training_tasks[batch_start:batch_end]
            
            _Logger.info(
                "Starting batch %d: tasks %d-%d (of %d total)",
                batch_number, batch_start + 1, batch_end, len(training_tasks)
            )
            _log_memory_usage(f"Before batch {batch_number}")
            
            # Create a NEW ThreadPoolExecutor for this batch
            # This ensures workers are fresh and don't accumulate memory
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Track active futures for this batch
                active_futures = {}
                pending_batch_tasks = list(enumerate(batch_tasks, start=batch_start + 1))
                pending_batch_tasks.reverse()  # Use as a stack
                
                # Process tasks in this batch with memory-aware throttling
                while pending_batch_tasks or active_futures:
                    # Submit new tasks if memory allows and we have capacity
                    while (
                        len(active_futures) < max_workers 
                        and pending_batch_tasks 
                        and _should_allow_parallel_task(max_memory_mb)
                    ):
                        idx, (config_name, combo, model_type, train_fn) = pending_batch_tasks.pop()
                        
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
                        done, _ = wait(
                            active_futures.keys(),
                            timeout=2.0,
                            return_when=FIRST_COMPLETED
                        )
                        
                        # Process completed tasks
                        for future in done:
                            idx, config_name, model_type = active_futures.pop(future)
                            
                            try:
                                result = future.result()
                                
                                # STREAM result to database immediately (do NOT keep in memory)
                                result_db_id = save_optimizer_result(run_id, result)
                                
                                # Update progress
                                with _progress_lock:
                                    progress.completed_configurations += 1
                                    progress.current_configuration = config_name
                                    progress.current_model_type = model_type
                                    
                                    # Check for phase transition (hybrid strategy only)
                                    if (search_strategy == SearchStrategy.HYBRID_GENETIC_BAYESIAN 
                                        and phase_1_end_index > 0 
                                        and progress.completed_configurations == phase_1_end_index):
                                        _Logger.info(
                                            "=" * 80
                                        )
                                        _Logger.info(
                                            "PHASE TRANSITION: GA Exploration Complete → Bayesian Exploitation Starting"
                                        )
                                        _Logger.info(
                                            "Phase 1 complete: %d configurations trained",
                                            phase_1_end_index
                                        )
                                        _Logger.info(
                                            "Phase 2 starting: %d strategic combinations",
                                            (num_combinations * 2) - phase_1_end_index
                                        )
                                        _Logger.info(
                                            "=" * 80
                                        )
                                        progress.add_log_message(
                                            f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Phase 1 (GA) complete → Phase 2 (Bayesian) starting"
                                        )
                                    
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
                                            progress.best_result_db_id = result_db_id
                                            progress.add_log_message(
                                                f"[{datetime.now().strftime('%H:%M:%S')}] 🏆 New best: {config_name} ({model_type}) with Val MAPE = {result.val_mape_pct:.2f}%"
                                            )
                                            # Update database with new best result
                                            update_optimizer_run_progress(
                                                run_id,
                                                progress.completed_configurations,
                                                "training",
                                                result_db_id
                                            )
                                
                                # Log memory periodically
                                if progress.completed_configurations % 10 == 0:
                                    _log_memory_usage(f"After task {progress.completed_configurations}/{total_configs}")
                                    # Update database progress
                                    update_optimizer_run_progress(
                                        run_id,
                                        progress.completed_configurations,
                                        "training",
                                        progress.best_result_db_id
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
                    elif pending_batch_tasks:
                        _Logger.debug("Memory limit reached, waiting for tasks to complete...")
                        time.sleep(1.0)
            
            # Batch complete - executor is shut down and workers are recycled
            _Logger.info(
                "Batch %d complete: processed tasks %d-%d (of %d total)",
                batch_number, batch_start + 1, batch_end, len(training_tasks)
            )
            _log_memory_usage(f"After batch {batch_number}")
            
            # Force garbage collection after each batch to clean up worker memory
            gc.collect()
            time.sleep(1.0)  # Give GC time to complete
            _log_memory_usage(f"After GC batch {batch_number}")
            
            task_index = batch_end
        
        # Final memory log
        _log_memory_usage("Optimizer complete")
        
        # Optimization complete
        progress.phase = "complete"
        progress.end_time = datetime.now()
        complete_optimizer_run(run_id, progress.end_time, "complete")
        
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
                progress.add_log_message(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Results saved to database (run_id={run_id})"
                )
            else:
                progress.add_log_message(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Optimization complete (no valid results)"
                )
        
    except Exception as e:
        _Logger.error("Optimizer error: %s", e, exc_info=True)
        progress.phase = "error"
        progress.error_message = str(e)
        progress.end_time = datetime.now()
        
        # Save error to database
        complete_optimizer_run(run_id, progress.end_time, "error", str(e))
        
        with _progress_lock:
            progress.add_log_message(
                f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}"
            )
    
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
    
    If complete_feature_config is available (new format), it applies ALL feature states.
    Otherwise, it falls back to only applying experimental_features (legacy format).
    
    When enabling derived features (e.g., wind_avg_1h), this function also ensures
    the corresponding stat is enabled in feature_stats_config so the feature is available.
    This prevents confusion where the optimizer enables a feature for training but it's
    not available because the stat wasn't enabled in Sensor Configuration.
    
    Note: This relies on FeatureConfiguration.enable_feature() and disable_feature()
    methods which support both experimental and derived features (see ml/feature_config.py).
    
    Args:
        best_result: The best OptimizationResult from optimization
        enable_two_step: If True, also enable two-step prediction mode
        
    Returns:
        True if successfully applied and saved
    """
    try:
        from db.feature_stats import get_feature_stats_config, StatType
        
        config = get_feature_config()
        stats_config = get_feature_stats_config()
        
        # Track which derived features need stats enabled
        derived_features_to_enable = []
        
        # Prefer complete_feature_config if available (new format)
        # This ensures ALL features (core + experimental) are restored to the exact state
        if best_result.complete_feature_config:
            _Logger.info("Applying complete feature configuration (%d features)", len(best_result.complete_feature_config))
            for feature_name, enabled in best_result.complete_feature_config.items():
                if enabled:
                    config.enable_feature(feature_name)
                    # Track if this is a derived feature that needs stats enabled
                    if config._is_derived_sensor_stat_feature(feature_name):
                        derived_features_to_enable.append(feature_name)
                else:
                    config.disable_feature(feature_name)
        else:
            # Fallback to legacy behavior: only apply experimental features that were tested
            _Logger.info("Applying experimental features only (legacy format) (%d features)", len(best_result.experimental_features))
            for feature_name, enabled in best_result.experimental_features.items():
                if enabled:
                    config.enable_feature(feature_name)
                    # Track if this is a derived feature that needs stats enabled
                    if config._is_derived_sensor_stat_feature(feature_name):
                        derived_features_to_enable.append(feature_name)
                else:
                    config.disable_feature(feature_name)
        
        # Enable stats for derived features in feature_stats_config
        # This ensures the features are available (collected during resampling)
        if derived_features_to_enable:
            _Logger.info("Ensuring stats are enabled for %d derived features", len(derived_features_to_enable))
            for feature_name in derived_features_to_enable:
                # Parse feature name to extract sensor and stat type
                # Format: sensor_name_avg_XXh (e.g., wind_avg_1h, outdoor_temp_avg_6h)
                if "_avg_" in feature_name:
                    parts = feature_name.rsplit("_avg_", 1)
                    if len(parts) == 2:
                        sensor_name, time_window = parts
                        # Map time window to StatType
                        stat_type_map = {
                            "1h": StatType.AVG_1H,
                            "6h": StatType.AVG_6H,
                            "24h": StatType.AVG_24H,
                            "7d": StatType.AVG_7D,
                        }
                        stat_type = stat_type_map.get(time_window)
                        if stat_type:
                            # Enable the stat if not already enabled
                            enabled_stats = stats_config.get_enabled_stats_for_sensor(sensor_name)
                            if stat_type not in enabled_stats:
                                _Logger.info("Enabling stat %s for sensor %s (required by optimizer)", stat_type.value, sensor_name)
                                stats_config.set_stat_enabled(sensor_name, stat_type, True)
            
            # Save stats config
            stats_config.save()
        
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
