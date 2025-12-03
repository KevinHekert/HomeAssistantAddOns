#!/usr/bin/env python3
"""
Simple integration test for Genetic Algorithm optimizer.

This script tests the complete flow without requiring pytest or external dependencies.
"""

import sys
sys.path.insert(0, '.')

print("=" * 60)
print("INTEGRATION TEST: Genetic Algorithm Optimizer")
print("=" * 60)

# Test 1: Import modules
print("\n[TEST 1] Importing modules...")
try:
    from ml.optimizer import (
        SearchStrategy,
        _generate_genetic_algorithm_combinations,
        _generate_hybrid_genetic_bayesian_combinations,
    )
    from ml.feature_config import EXPERIMENTAL_FEATURES
    print("✓ Successfully imported SearchStrategy and generator functions")
    print(f"  Available strategies: {[s.value for s in SearchStrategy]}")
    print(f"  EXPERIMENTAL_FEATURES count: {len(EXPERIMENTAL_FEATURES)}")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Generate GA combinations
print("\n[TEST 2] Generating Genetic Algorithm combinations...")
try:
    combos_gen = _generate_genetic_algorithm_combinations(
        include_derived=False,
        population_size=5,
        num_generations=3,
        mutation_rate=0.1,
    )
    combos = list(combos_gen)
    expected = 5 * 3
    
    assert len(combos) == expected, f"Expected {expected} combinations, got {len(combos)}"
    assert all(isinstance(c, dict) for c in combos), "All combinations should be dicts"
    
    # Check first is baseline
    assert all(not v for v in combos[0].values()), "First should be baseline"
    
    print(f"✓ Generated {len(combos)} combinations successfully")
    print(f"  Population: 5, Generations: 3")
    print(f"  First combination (baseline): {sum(combos[0].values())} features enabled")
    print(f"  Last combination: {sum(combos[-1].values())} features enabled")
except Exception as e:
    print(f"✗ GA generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Generate Hybrid combinations
print("\n[TEST 3] Generating Hybrid GA + Bayesian combinations...")
try:
    combos_gen = _generate_hybrid_genetic_bayesian_combinations(
        include_derived=False,
        ga_population_size=4,
        ga_num_generations=2,
        bayesian_iterations=3,
    )
    combos = list(combos_gen)
    expected = (4 * 2) + 3  # GA phase + Bayesian phase
    
    assert len(combos) == expected, f"Expected {expected} combinations, got {len(combos)}"
    
    print(f"✓ Generated {len(combos)} combinations successfully")
    print(f"  GA phase: 4 pop × 2 gen = 8 combinations")
    print(f"  Bayesian phase: 3 combinations")
    print(f"  Total: {len(combos)} combinations")
except Exception as e:
    print(f"✗ Hybrid generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check diversity
print("\n[TEST 4] Checking combination diversity...")
try:
    combos_gen = _generate_genetic_algorithm_combinations(
        include_derived=False,
        population_size=20,
        num_generations=1,
    )
    combos = list(combos_gen)
    
    # Count unique combinations
    unique_combos = set(
        frozenset(k for k, v in combo.items() if v)
        for combo in combos
    )
    
    diversity_ratio = len(unique_combos) / len(combos)
    
    print(f"✓ Diversity check passed")
    print(f"  Total combinations: {len(combos)}")
    print(f"  Unique combinations: {len(unique_combos)}")
    print(f"  Diversity ratio: {diversity_ratio:.1%}")
    
    assert len(unique_combos) >= 10, f"Expected at least 10 unique combinations"
except Exception as e:
    print(f"✗ Diversity check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Feature count validation
print("\n[TEST 5] Validating feature counts...")
try:
    combos_gen = _generate_genetic_algorithm_combinations(
        include_derived=False,
        population_size=10,
        num_generations=1,
    )
    combos = list(combos_gen)
    
    n_features = len(EXPERIMENTAL_FEATURES)
    
    # All combinations should have exactly n_features keys
    for i, combo in enumerate(combos):
        assert len(combo) == n_features, f"Combo {i}: expected {n_features} features, got {len(combo)}"
    
    # Count how many features are enabled in each combination
    feature_counts = [sum(1 for v in combo.values() if v) for combo in combos]
    
    print(f"✓ Feature count validation passed")
    print(f"  All combinations have {n_features} features")
    print(f"  Enabled features range: {min(feature_counts)} to {max(feature_counts)}")
    print(f"  Average enabled: {sum(feature_counts) / len(feature_counts):.1f}")
except Exception as e:
    print(f"✗ Feature count validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("\nSummary:")
print("  - SearchStrategy enum works")
print("  - Genetic Algorithm generator works")
print("  - Hybrid GA + Bayesian generator works")
print("  - Combinations are diverse and valid")
print("  - Feature counts are correct")
print("\nThe optimizer is ready to handle ANY number of features!")
print("Example: 52 features → 5,100 combinations × 2 models = 10,200 trainings")
print("         (vs 2^52 = 4.5 quadrillion exhaustive)")
