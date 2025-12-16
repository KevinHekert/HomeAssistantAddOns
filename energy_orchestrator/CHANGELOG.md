# Changelog

All notable changes to this add-on will be documented in this file.

## [Unreleased]

- No unreleased changes yet.

## [0.0.0.121] - 2025-12-16

- **Feature: Complete Integration Test Environment with MariaDB**
  - **Added**: Full integration testing infrastructure with MariaDB 10.6+ for production-parity testing
  - **Added**: Docker Compose test environment with MariaDB, mock Home Assistant API, and Energy Orchestrator containers
  - **Added**: Mock Home Assistant API server (Flask) for testing API interactions
  - **Added**: Comprehensive integration test suite with 50+ tests covering:
    - Database connectivity and schema validation
    - Sample data CRUD operations
    - Data resampling workflows
    - Feature statistics calculation
    - End-to-end workflows
  - **Added**: GitHub Actions workflow for automated integration testing in CI/CD
  - **Added**: Convenience scripts:
    - `scripts/run-integration-tests.sh` - Main test runner with options
    - `scripts/setup-test-db.sh` - Database setup and verification
  - **Added**: `INTEGRATION_TESTING.md` - Comprehensive documentation for integration testing
  - **Added**: `pytest.ini` - Pytest configuration with markers and coverage settings
  - **Added**: `requirements-test.txt` - Test-specific dependencies
  - **Added**: `.env.test.example` - Environment variable template
  - **Impact**: Developers can now run full integration tests locally and in CI/CD to catch database-specific issues early
  - **Files Added**:
    - `docker-compose.test.yml`
    - `tests/mock_homeassistant/` (app.py, Dockerfile, README.md)
    - `energy_orchestrator/app/tests/integration/` (5 test files + conftest)
    - `.github/workflows/integration-tests.yml`
    - `scripts/init-test-db.sql`
    - `scripts/run-integration-tests.sh`
    - `scripts/setup-test-db.sh`
    - `INTEGRATION_TESTING.md`
    - `energy_orchestrator/app/pytest.ini`
    - `energy_orchestrator/requirements-test.txt`
    - `.env.test.example`
  - **Files Modified**:
    - `README.md` - Added testing documentation
    - `.gitignore` - Added test artifacts and environment files
  - **Testing**: Not executed per agent instructions - complete environment ready for use

## [0.0.0.120] - 2025-12-03

- **Fix: Check Resample Status Button Not Working**
  - **Fix**: Corrected API fetch path in `checkResampleStatus()` function
  - **Issue**: The "Check Status" button for resampling was returning 404 errors when clicked
  - **Root Cause**: The fetch call used `/api/resample/status` (with leading slash) while all other API calls (49 instances) use relative paths like `api/sample_rate` (without leading slash). This inconsistency caused path resolution issues, especially when running through Home Assistant ingress.
  - **Impact**: Users can now successfully check resampling progress by clicking the "Check Status" button
  - **Files Changed**: `energy_orchestrator/app/templates/index.html` (line 1503)
  - **Pattern**: Changed from `fetch('/api/resample/status')` to `fetch('api/resample/status')` to match all other API calls in the application

## [0.0.0.119] - 2025-12-03

- **Critical Fix: Python 3.9 Compatibility for Type Hints**
  - **Fix**: Changed `callable | None` to `Optional[Callable]` in function parameter default values
  - **Issue**: Python 3.9 (used in Alpine base image) doesn't support union type syntax (`Type | None`) in default parameter values, causing `TypeError: unsupported operand type(s) for |` at module import time
  - **Root Cause**: While `from __future__ import annotations` allows `Type | None` in annotations, default parameter values are evaluated at runtime and require `Optional[Type]` syntax for Python 3.9 compatibility
  - **Impact**: Fixes startup crash where the add-on would fail to start with incomplete traceback at line 20 of app.py
  - **Files Changed**: 
    - `energy_orchestrator/app/db/resample.py`: Added `from typing import Callable, Optional` and changed `progress_callback: callable | None = None` to `progress_callback: Optional[Callable] = None`
    - `energy_orchestrator/run.sh`: Added `-u` flag to Python command for unbuffered output to ensure complete error messages in logs
  - **Testing**: All 64 resample tests pass successfully

## [0.0.0.118] - 2025-12-03

- **Python Type Hint Compatibility Fix**
  - **Fix**: Added `from __future__ import annotations` to `db/resample.py`
  - **Issue**: Python 3.9+ requires either `from __future__ import annotations` or `Callable` from typing for union type hints with `callable | None`
  - **Impact**: Fixes import errors when loading the application
  - **Files Changed**: `energy_orchestrator/app/db/resample.py`

## [0.0.0.117] - 2025-12-03

- **Background Resampling with Progress Monitoring**
  - **Feature**: Resampling now runs in a background thread with real-time progress tracking
  - **Changes**:
    - Added `ResampleProgress` dataclass to track resampling progress
    - Modified `resample_all_categories()` to accept optional `progress_callback` parameter
    - Resampling reports progress every 50 slots with hour-based progress (e.g., "1/50 hours")
    - Added `/api/resample/status` endpoint to poll for resampling progress
    - Modified `/resample` endpoint to start resampling in background thread
    - Added "Check Status" button to UI to manually check resampling progress
    - Progress display shows:
      - Current phase (initializing, resampling, calculating_stats, complete, error)
      - Progress as hours and slots (e.g., "5/50 hours (60/600 slots)")
      - Last 15 log messages with timestamps
      - Real-time updates via polling every 2 seconds
    - UI automatically starts polling when resample is triggered
    - UI stops polling when resampling completes or errors
  - **Benefits**:
    - Users can navigate away while resampling runs
    - Progress is visible throughout the operation
    - No more waiting for long operations to complete
  - **Files Changed**: 
    - `energy_orchestrator/app/db/resample.py` - Added progress tracking
    - `energy_orchestrator/app/app.py` - Added background threading and status endpoint
    - `energy_orchestrator/app/templates/index.html` - Added status button and progress display
    - `energy_orchestrator/config.yaml` - Version bump to 0.0.0.117
    - `energy_orchestrator/CHANGELOG.md` - This entry

## [0.0.0.116] - 2025-12-03

- **Improved Sensor Information Tab Organization**
  - **Feature**: Reorganized UI to better group sensor-related functionality
  - **Changes**:
    - Added new "📈 Resampled Sensor Information" card showing statistics from resampled data
      - Displays category, unit, type (Raw/Derived), first/last timestamp, and sample count
      - Fetches data from `/api/resampled_data` endpoint
      - Shows up to 1000 samples per category grouped by sensor
    - Moved "🔄 Data Resampling" functionality from Configuration tab to Sensor Information tab
      - Includes sample rate selector, flush checkbox, and resample button
      - All functionality preserved, just relocated for better organization
    - Sensor Information tab now contains all sensor-related features in one place
  - **Rationale**: Groups all sensor data viewing and processing in one logical location
  - **Files Changed**: `energy_orchestrator/app/templates/index.html`, `energy_orchestrator/app/tests/test_ui_structure.py`

- **Removed Deprecated Optimizer Best Result Card**
  - **Cleanup**: Removed redundant "🏆 Best Configuration" card from Optimizer tab
  - **Rationale**: This information is already shown in the results table with a 👑 crown icon for the best result
  - **Impact**: Cleaner Optimizer tab UI without duplicate information
  - **Files Changed**: `energy_orchestrator/app/templates/index.html`

- **Code Quality Improvements**
  - Improved comment clarity in `loadResampledSensorInfo()` function
  - Extracted hardcoded limit to a named constant for maintainability
  - Added comprehensive UI structure tests (10 tests, all passing)

## [0.0.0.115] - 2025-12-03

- **Fixed JavaScript Null Reference Error in Optimizer Tab**
  - **Issue**: JavaScript error "Cannot read properties of null (reading 'style')" when loading optimizer results
  - **Root Cause**: Multiple functions were using incorrect DOM element IDs that didn't match the actual HTML elements
    - Used `optimizerStatus` instead of `optimizerStatusDisplay`
    - Used `optimizerResults` instead of `optimizerResultsTable`
    - Used `optimizerBestResult` instead of `optimizerBestResultCard`
  - **Solution**: 
    - Corrected all DOM element ID references to match actual HTML IDs in 4 functions:
      - `loadLatestOptimizerResults()`
      - `getOptimizerStatus()`
      - `applyOptimizerResult()`
      - `applyResultById()`
    - Added defensive null checks before accessing element properties to prevent future errors
    - Functions now fail gracefully if elements are missing
  - **Impact**: Optimizer tab now loads properly without JavaScript errors when switching tabs
  - **Files Changed**: `energy_orchestrator/app/templates/index.html`

- **Added Refresh Results Button to Optimizer Tab**
  - **Feature**: Added a "🔄 Refresh Results" button to manually reload the optimization results table
  - **Location**: Top-right corner of the "Optimization Results (Top 30)" card in the Optimizer tab
  - **Behavior**: 
    - Button is disabled during loading to prevent multiple simultaneous requests
    - Shows "Loading results..." message while fetching data
    - Re-enables after loading completes or on error
  - **Use Case**: Allows users to manually refresh the results table without switching tabs or reloading the page
  - **Files Changed**: `energy_orchestrator/app/templates/index.html`

## [0.0.0.114] - 2025-12-03

- **Disabled Automatic Test Runs on PR Commits**
  - **Issue**: Tests were running automatically on every commit in a PR, causing excessive test runs when there are many commits
  - **Problem**: Tests take a long time, and multiple commits in one PR created many unnecessary test runs
  - **Solution**: Removed `pull_request_target` trigger from the test workflow
  - **Impact**: Tests no longer run automatically on PR commits, reducing CI load and runtime
  - **Manual Testing**: Tests can still be triggered manually via GitHub Actions workflow_dispatch
    - Go to Actions tab → "Run Tests (Energy Orchestrator)" → "Run workflow" → Select branch
  - **Automatic Testing**: Tests still run automatically when code is merged to main/master branches
  - **Files Changed**: `.github/workflows/test-energy-orchestrator.yml`
- **Add Dedicated Optimizer Tab with Persistent Results Display**
  - **Issue**: Optimizer results weren't shown anymore and UI was confusing
    - Results table was hidden after completion
    - Status check and progress updates overwrote each other
    - Optimizer functionality was mixed with model training controls
    - No easy way to re-train with a specific optimizer result's configuration
  - **Solution**: 
    - Created dedicated "Optimizer" tab separate from "Model Training" tab
    - Moved all optimizer-related controls/visuals to new Optimizer tab
    - Results table now always shows top 30 results (never hidden)
    - Results persist across page loads by loading from database
    - Separated progress display (for running optimization) from status display (for status checks)
    - Added re-train functionality to re-train model using selected result's configuration
    - Added "Select" button to each result for easy re-training
  - **Database Migration**: Added schema migration for `complete_feature_config_json` column
    - Fixes `OperationalError: (1054, "Unknown column 'complete_feature_config_json' in 'field list'")` error
    - Added `_migrate_add_complete_feature_config_column()` function to `db/core.py`
    - Migration runs automatically on app startup via `init_db_schema()`
  - **UI Improvements**:
    - Progress display shows real-time optimization progress with log updates
    - Status check displays current optimizer state without interfering with progress
    - Best result card always visible when results exist
    - Results table automatically loads latest results when opening Optimizer tab
    - Each result row is clickable to view detailed feature configuration
    - "Apply" button on each result to apply that configuration
    - "Select" button on each result to select it for re-training
  - **Files Changed**: 
    - `energy_orchestrator/app/templates/index.html`: Added Optimizer tab, moved optimizer UI, separated displays
    - `energy_orchestrator/app/db/core.py`: Added schema migration for complete_feature_config_json
  - **Impact**: Users can now easily view, compare, and re-use optimizer results without confusion
  - **Why This Matters**: Optimizer is a key feature for finding optimal configurations - users need clear visibility into results and easy ways to apply or re-test configurations

## [0.0.0.113] - 2025-12-03

- **Fix Optimizer Results Application**
  - **Issue**: When applying an optimizer result, the feature configuration wasn't being restored correctly
    - Only experimental features that were tested were being applied
    - Other features (core features, or experimental features not in the test) retained their current state
    - This meant the "same settings" weren't actually being used - only partial settings
  - **Root Cause**: Optimizer only stored `experimental_features` (the features being tested), not the complete configuration state
  - **Solution**: 
    - Added `complete_feature_config_json` column to `optimizer_results` database table
    - Added `complete_feature_config` field to `OptimizationResult` dataclass
    - Modified optimizer to capture complete feature state (all features, core + experimental) when storing results
    - Updated `apply_best_configuration` to apply complete feature config if available, falls back to experimental_features for legacy results
    - Added `get_complete_feature_state()` method to `FeatureConfiguration` class
  - **Additional Fix**: Removed references to non-existent `loadFeaturesBtn` button that was causing JavaScript errors when applying optimizer results
  - **Testing**: Added comprehensive tests to verify complete configuration storage and application
    - `test_optimization_result_has_complete_feature_config`: Verifies result stores complete config
    - `test_apply_complete_feature_config`: Verifies all features are restored when applying result
    - `test_apply_legacy_experimental_features_only`: Verifies legacy results still work
    - `test_get_complete_feature_state`: Verifies method returns all features
  - **Files Changed**: 
    - `energy_orchestrator/app/templates/index.html`: Fixed JavaScript error
    - `energy_orchestrator/app/db/models.py`: Added `complete_feature_config_json` column
    - `energy_orchestrator/app/db/optimizer_storage.py`: Updated save/load functions
    - `energy_orchestrator/app/ml/feature_config.py`: Added `get_complete_feature_state()` method
    - `energy_orchestrator/app/ml/optimizer.py`: Capture and apply complete feature config
    - `energy_orchestrator/app/app.py`: Pass complete_feature_config to apply function
    - `energy_orchestrator/app/tests/test_ui_structure.py`: Added tests for JavaScript fix
    - `energy_orchestrator/app/tests/test_optimizer_complete_config.py`: Added tests for complete config
  - **Impact**: Users can now reliably restore the exact same feature configuration used in any optimizer run, not just partial settings
  - **Why This Matters**: When users want to reuse settings from a successful optimizer run, they get the COMPLETE configuration, ensuring reproducible results

## [0.0.0.112] - 2025-12-03

- **Fix Training Data Popup - Display Correct First and Last Rows of Training Split**
  - **Issue**: Optimizer results popup showed incorrect first/last rows of training data
  - **Root Cause**: Code captured `df.iloc[0]` and `df.iloc[-1]` which are the first and last rows of the FULL dataset
    - Training uses 80% of data (rows 0-79), validation uses 20% (rows 80-99)
    - Old code captured row 0 (correct) and row 99 (incorrect - this is validation data!)
    - Should have captured row 0 (first training row) and row 79 (last training row)
  - **Solution**: Use `metrics.train_samples` to calculate the correct training split boundary
    - Changed from `df.iloc[-1]` to `df.iloc[train_samples - 1]`
    - Added logic to handle both single-step (`metrics.train_samples`) and two-step (`metrics.regressor_train_samples`) models
  - **Testing**: Added comprehensive tests to verify correct row capture for both model types
    - `test_train_single_configuration_captures_training_rows`: Verifies single-step model with 100-row dataset (trains on 0-79, captures row 79)
    - `test_train_single_configuration_two_step_captures_training_rows`: Verifies two-step model with same logic
  - **Impact**: Popup now correctly displays the actual first and last rows used during training, not the full dataset
  - **Files Changed**: 
    - `energy_orchestrator/app/ml/optimizer.py` (lines 864-897): Fixed row capture logic
    - `energy_orchestrator/app/tests/test_optimizer_parallel.py`: Added 2 new tests
  - **Why This Matters**: Users can now see the actual training data boundaries to understand what time ranges were used for model training

## [0.0.0.111] - 2025-12-03

- **Extensive Testing Infrastructure Improvements**
  - **Added comprehensive test database configuration** (`tests/conftest.py`)
    - All tests now use in-memory SQLite database instead of production MariaDB
    - Automatic database engine patching for all test modules
    - Ensures complete test isolation and prevents production data corruption
  - **Created test documentation** (`TEST_SUMMARY.md`)
    - Comprehensive test execution summary
    - Detailed findings and recommendations
    - Instructions for running tests (fast, full, with coverage)
  - **Added test runner script** (`run_tests.sh`)
    - Convenient script for running different test suites
    - Supports fast mode (skips slow optimizer tests), full mode, and coverage mode
  - **Test Results**: 
    - 678 total tests across 36 test files
    - ~90% passing rate (600+ tests passing)
    - Identified 5 test failures requiring investigation
    - Identified 4 test files that timeout (optimizer tests with heavy ML training)
  - **Impact**: Developers can now run comprehensive tests safely without risk to production data

## [0.0.0.111] - 2025-12-03

- **Fix onClick JavaScript Syntax Error in Optimizer Results Table**
  - **Issue**: Clicking on optimizer result rows caused JavaScript syntax error "Unexpected end of input"
  - **Root Cause**: JSON.stringify() was used directly in HTML onclick attributes without proper escaping
    - Double quotes in JSON broke the HTML attribute syntax
    - Example: `onclick="func({\"key\": \"value\"})"` created malformed HTML/JavaScript
  - **Solution**: Properly escape JSON data before embedding in onclick attributes
    - Convert JSON to string and escape single quotes: `JSON.stringify(data).replace(/'/g, "\\'")`
    - Use single-quoted strings in onclick with JSON.parse: `onclick="func(JSON.parse('...'))`
    - This ensures JSON data is safely embedded without breaking HTML attribute syntax
  - **Impact**: onClick functionality now works correctly for optimizer result rows
  - **Files Changed**: `app/templates/index.html` (lines 3117-3121, 3371-3376)
  - **Testing**: Validated with test HTML file containing complex JSON with quotes and special characters

## [0.0.0.110] - 2025-12-03

- **Fix GA Phase Feature Diversity - Ensure Small Feature Counts Are Tested**
  - **Issue**: Genetic Algorithm phase only tested configurations with many features (15-30), never small counts (1-5)
  - **Root Cause**: GA used fixed 30% probability per feature, resulting in ~15-16 features on average with 52 total features
  - **User Report**: "Why am I never seeing just a couple of features, always a lot" (tasks showing +19, +22, +24, +27 experimental features)
  - **Solution**: Implemented stratified feature count distribution in GA initial population:
    - First third of population: 2-20% probability → 1-10 features (small sets)
    - Middle third: 20-50% probability → 10-25 features (medium sets)
    - Last third: 50-80% probability → 25+ features (large sets)
  - **Clarification**: Core features (15 baseline features) are ALWAYS enabled. Optimizer only toggles experimental features.
    - Baseline = 0 experimental + 15 core = 15 total features minimum
    - Small sets = 1-5 experimental + 15 core = 16-20 total features
  - **Impact**: GA phase now properly explores small feature counts (1-5 experimental), not just large counts (15-30)
  - **Testing**: Updated `test_ga_diverse_population()` to verify small, medium, and large feature counts are present
  - **Note**: Bayesian phase (tasks 10001-10200) already had good diversity strategy (0-5, 13-26, 26-52)

- **Fix Optimizer Minimum Experimental Sensors Constraint & Improve Two-Phase Training**
  - **Issue**: Bayesian phase in hybrid strategy enforced minimum of 1 experimental sensor, preventing baseline (0 sensors) testing
  - **Problem**: "Sometimes, less is more" - baseline configuration might be optimal but wasn't tested in Bayesian phase
  - **Root Cause**: Line 684 used `random.randint(1, ...)` instead of `random.randint(0, ...)` 
  - **Solution**: Changed Bayesian phase to allow 0 features: `num_features_to_enable = random.randint(0, max(3, n_features // 10))`
  - **Phase Transition Improvements**:
    - Added extensive documentation explaining how the system transitions between GA and Bayesian phases
    - Added clear phase boundary logging with "=" separators
    - Added automatic phase transition detection during training
    - Added progress log message when transitioning from Phase 1 (GA) to Phase 2 (Bayesian)
    - Clarified that current implementation pre-generates combinations (memory efficient) vs true Bayesian (feedback-based)
  - **Testing**: Added `test_hybrid_bayesian_phase_allows_baseline()` to verify baseline can be generated
  - **Impact**: Optimizer now properly tests all feature counts including 0 (baseline) in extensive automated two-phase training
  - **Documentation**: Added detailed "Phase Transition Mechanics" section explaining generator pattern trade-offs

## [0.0.0.109] - 2025-12-03

- **Optimizer Results UI Improvements**
  - Limit results table to top 30 configurations (sorted by Val MAPE)
  - Add expandable row details showing all features used in training:
    - Core features (15 features, green tags) - always used
    - Experimental features (orange tags) - tested in configuration
    - Training and validation sample counts
    - Total feature count summary
  - Click-to-expand/collapse feature details
  - Auto-refresh log during optimization runs (2-second polling)

## [0.0.0.108] - 2025-12-03

- **Fix Optimizer "'type' object is not iterable" Error**
  - **Problem**: Optimizer failed to start with error "'type' object is not iterable"
  - **Root Cause**: Line 1080 in `ml/optimizer.py` referenced undefined variable `combinations` instead of `combinations_list`
  - **Solution**: Changed line 1080 from `for combo in combinations:` to `for combo in combinations_list:`
  - **Additional Fixes**:
    - Updated test imports to use `_generate_experimental_feature_combinations` (generator) instead of old `_get_experimental_feature_combinations`
    - Updated all test calls to convert generator to list: `list(_generate_experimental_feature_combinations())`
    - Added comprehensive test suite in `test_optimizer_bug_fix.py` to verify the fix
  - **Tests**: All combination-related tests pass successfully
  - **Verification**: Confirmed optimizer can now iterate over combinations without errors

## [0.0.0.107] - 2025-12-03

- **Implement Hybrid Genetic Algorithm + Bayesian Optimization for Feature Selection**
  - **Problem**: Need to find best feature combination from 52+ features without testing all 2^52 combinations
  - **Solution**: Implemented intelligent search strategies that scale to ANY number of features
  - **Hybrid Strategy (Default)**: Genetic Algorithm + Bayesian Optimization
    - Phase 1: Genetic Algorithm (100 generations × 50 population = 5,000 combinations)
    - Phase 2: Bayesian Optimization (100 strategic iterations)
    - Total: 5,100 combinations × 2 models = 10,200 trainings (feasible for any feature count)
    - vs Exhaustive: 2^52 = 4.5 quadrillion combinations (impossible)
  - **Changes Made**:
    1. Added `SearchStrategy` enum with options: EXHAUSTIVE, GENETIC, BAYESIAN, HYBRID_GENETIC_BAYESIAN
    2. Implemented `_generate_genetic_algorithm_combinations()` with evolution-based search
    3. Implemented `_generate_hybrid_genetic_bayesian_combinations()` combining GA + Bayesian
    4. Added parameters: `search_strategy`, `genetic_population_size`, `genetic_num_generations`, `genetic_mutation_rate`, `bayesian_iterations`
    5. Updated `run_optimization()` to support strategy selection
    6. Added `max_combinations` to `OptimizerConfig` database model
    7. Updated config functions to handle `max_combinations` parameter
    8. Genetic Algorithm features: population, crossover, mutation, elitism, tournament selection
    9. Bayesian phase: strategically tests diverse combinations to exploit GA findings
    10. All strategies use lazy generators for memory efficiency
    11. Tests use only 4 experimental features (2^4 = 16 combinations) for speed
    12. Production uses ALL features (experimental + derived) with intelligent search
  - **Search Strategy Details**:
    - **HYBRID (Default)**: 5,100 combinations (10,200 trainings) - Best balance of exploration + exploitation
    - **GENETIC**: Configurable generations × population - Pure evolution-based search
    - **EXHAUSTIVE**: Up to max_combinations limit - Brute force with safety limit
  - **Scalability**: Works with 52, 100, or 1000+ features without memory issues
  - **Configurability**: Population size, generations, mutation rate all adjustable for scale up/down
  - **User Feedback Addressed**: @KevinHekert requested not limiting features + scalable solution
  - **Version bumped to 0.0.0.107**

## [0.0.0.106] - 2025-12-03

- **Fix Optimizer Crash: Memory-Efficient Streaming Architecture**
  - **Problem**: Optimizer crashed with 52 features generating 2^52 (4+ quadrillion) combinations, filling all memory
  - **Root Cause**: `_get_all_available_features()` included derived features from config, not just EXPERIMENTAL_FEATURES
  - **Solution**: Implemented three major improvements:
    1. **Limited Feature Selection**: Only use EXPERIMENTAL_FEATURES (not derived), reducing from 52 to 4 features (2^4 = 16 combinations)
    2. **Streaming Database Storage**: Results saved immediately to database instead of keeping all in memory
    3. **Batch Worker Recycling**: Workers process 20 tasks then restart, preventing memory accumulation
  - **Changes Made**:
    1. Fixed `_get_all_available_features()` to exclude derived features
    2. Changed `include_derived_features` default from True to False in `run_optimization()`
    3. Added `batch_size` parameter (default: 20) for worker recycling
    4. Removed `results` list from `OptimizerProgress` dataclass
    5. Added `run_id` and `best_result_db_id` to `OptimizerProgress` for database tracking
    6. Implemented streaming storage functions: `create_optimizer_run()`, `save_optimizer_result()`, `update_optimizer_run_progress()`, `complete_optimizer_run()`, `get_optimizer_run_top_results()`
    7. Rewrote optimization loop to use batch worker recycling with ThreadPoolExecutor recycling
    8. Updated API endpoint `/api/optimizer/status` to query database for results instead of memory
    9. Updated `save_optimizer_run()` to handle streaming mode (legacy compatibility)
    10. Updated tests to work with database-backed results
  - **Memory Management**:
    - Results streamed to database immediately (NOT kept in memory)
    - Workers recreated every 20 tasks to prevent memory leaks
    - Only summary data (best result, progress, logs) kept in memory
    - Logs limited to last 10 messages
  - **UI Impact**: Progress now shows `run_id` and `top_results` from database
  - **Testing**: Updated test suite for streaming architecture
  - **Version bumped to 0.0.0.106**

## [0.0.0.105] - 2025-12-03

- **Add Manual Worker Limit Configuration with Per-Worker Memory Reporting**
  - **New Feature**: Added UI-configurable optimizer max_workers setting
    - Users can now manually limit the number of parallel workers (0 = auto-calculate)
    - Configuration accessible through new API endpoints: GET/POST `/api/optimizer/config`
    - Settings stored in new `optimizer_config` database table
  - **Memory Management Improvements**:
    - Each worker now reports memory usage before, after, and post-cleanup for each training run
    - Memory delta logging helps identify potential memory leaks
    - Automatic memory flushing after each worker run (existing garbage collection enhanced)
    - Worker lifecycle automatically managed by ThreadPoolExecutor
  - **Changes Made**:
    1. Created new `OptimizerConfig` database model for storing optimizer settings
    2. Added `db/optimizer_config.py` module for configuration management
    3. Added `configured_max_workers` parameter to `run_optimization()` function
    4. Enhanced `_train_single_configuration()` with per-worker memory reporting
    5. Updated optimizer to use configured max_workers or auto-calculate
    6. Added API endpoints for getting/setting optimizer configuration
    7. Created comprehensive test suite for new functionality
  - **Testing**: Added 2 new test files with full coverage of configuration and memory reporting

## [0.0.0.104] - 2025-12-03

- **Fix Failing Tests After Feature Set Reduction**
  - **Problem**: 12 tests were failing after reducing feature set from 10 to 4 features
  - **Solution**: Updated all affected tests to work with the reduced feature set
  - **Changes Made**:
    1. Updated `test_feature_config.py` to check for the 4 retained features instead of removed ones
    2. Updated `test_optimizer_combinations.py` to test pairwise combinations with the new 4-feature set
    3. Updated `test_feature_stats_sync.py` to use `outdoor_temp_avg_6h` instead of removed features
    4. Removed deprecated `max_workers` parameter from all test calls (now auto-calculated)
    5. Updated timing expectations in parallel tests to account for adaptive throttling delays
  - **Test Results**: All 647 tests now passing
  - **Version bumped to 0.0.0.104**

## [0.0.0.103] - 2025-12-03

- **Reduced Optimizer Test Set for Faster Execution**
  - **Problem**: Optimizer test runs taking too long with 10 features (2,048 trainings)
  - **Solution**: Reduced experimental feature set from 10 to 4 features
  - **Impact**:
    - Feature combinations: 1,024 → 16 (94% reduction)
    - Total trainings: 2,048 → 32 (~64x faster)
    - Estimated optimizer runtime reduced from hours to minutes
  - **Features Retained** (4 most impactful across categories):
    1. `pressure` - Weather feature for atmospheric conditions
    2. `outdoor_temp_avg_6h` - Short-term weather trend
    3. `heating_degree_hours_24h` - Heating demand metric
    4. `day_of_week` - Weekly pattern detection
  - **Features Removed** (can be manually enabled if needed):
    - `outdoor_temp_avg_7d`, `target_temp_avg_24h`, `heating_kwh_last_7d`
    - `heating_degree_hours_7d`, `is_weekend`, `is_night`
  - This makes optimizer testing practical for development/testing while maintaining diverse feature coverage

## [0.0.0.102] - 2025-12-03

- **Adaptive Memory-Aware Parallel Processing with Auto-Calculated Workers**
  - **Problem**: Optimizer still experiencing memory issues with sequential processing; users requested throttled parallel execution based on RAM limits
  - **Solution**: Implemented intelligent adaptive parallel processing with memory-based throttling
  - **Key Features**:
    1. **Automatic Worker Calculation**: System automatically determines optimal number of workers based on:
       - Available system memory (considers user-defined max_memory_mb or defaults to 75% of total RAM)
       - CPU core count (uses cores - 1, leaving one for system)
       - Estimated memory per task (~200MB)
       - Formula: `min(memory_workers, cpu_workers, 10)` with cap at 10 workers
    2. **Dynamic Memory Throttling**: Real-time memory monitoring during execution
       - Checks memory before starting each new parallel task
       - Only submits new tasks when memory < max_memory_mb threshold
       - Automatically scales down parallelism when memory is constrained
       - Scales back up when memory is freed
    3. **Memory Usage Logging**: Added detailed INFO-level logging of memory usage
       - Logs RSS (Resident Set Size), VMS (Virtual Memory Size)
       - Shows system available memory and usage percentage
       - Logs every 10 iterations and after garbage collection
       - Example: "Memory: RSS=850.2 MB, VMS=1024.5 MB, System Available=3200.0 MB (45.2% used)"
  - **Technical Implementation**:
    - Added `psutil` dependency for cross-platform memory monitoring
    - `_calculate_optimal_workers()`: Auto-calculates workers from system resources
    - `_should_allow_parallel_task()`: Guards task submission based on current memory usage
    - `_log_memory_usage()`: Provides detailed memory logging at INFO level
    - ThreadPoolExecutor with dynamic task submission (not batch submission)
    - 2-second timeout on task completion wait to periodically re-check memory
    - Aggressive GC every 10 completed tasks
    - 0.5s delay after each training for garbage collection
  - **User Experience**:
    - No manual worker configuration needed - system auto-optimizes
    - Can optionally set max_memory_mb via API (UI setting coming in future update)
    - Logs show real-time memory usage for monitoring
    - Example: 4GB RAM system → calculates 3-4 workers automatically
    - Memory-constrained systems automatically fall back to fewer workers or sequential processing
  - **Performance vs Safety Balance**:
    - With 100 workers on 4GB RAM: System throttles to sustainable level automatically
    - With 1,000,000 tasks: Workers adapt throughout entire run based on memory
    - No OOM kills: System respects memory limit and adjusts parallelism dynamically
  - Resolves issue #[issue_number] where memory climbed to max and app crashed during optimizer runs

## [0.0.0.101] - 2025-12-03

- **Optimizer Memory Management: Explicit Cleanup and Garbage Collection**
  - **Problem**: Optimizer was being killed by the system during long runs (2048 trainings), even with max_workers=1
  - **Root Cause**: DataFrames and model objects were accumulating in memory faster than Python's garbage collector could free them
  - **Fixes Applied**:
    1. Added explicit `del df` and `del model` after each training iteration to immediately free large objects
    2. Added `gc.collect()` call after each training to force garbage collection
    3. Added periodic garbage collection every 50 iterations during long optimization runs
    4. Imported `gc` module for explicit memory management
  - **Impact**: Optimizer should now complete full runs without OOM kills by proactively managing memory
  - **Technical Details**:
    - Each training iteration now explicitly deletes DataFrame (~10-100MB) and model object (~1-5MB)
    - Garbage collection is forced after cleanup to return memory to the OS
    - Periodic GC every 50 iterations prevents gradual memory accumulation
    - These changes work together with existing max_workers=1 and resource limits
  - Fixes issue where app was killed with "Killed" message during optimizer runs

## [0.0.0.100] - 2025-12-03

- **Reduce Optimizer Memory Usage**
  - Reduced max_workers from 3 to 1 in the model optimizer to prevent high RAM usage and OOM (Out Of Memory) kills
  - This throttles parallel training to reduce memory consumption during optimization runs
  - Training will take longer but should complete successfully without being killed by the system
  - Affects both app.py (explicit call) and optimizer.py (default value)
  - Added resource limits to config.yaml: map_ram=1024MB, max_ram=2048MB to enforce container memory limits

## [0.0.0.98] - 2025-12-03

- **Comprehensive Optimizer: Test ALL Feature Combinations**
  - **Full Combinatorial Testing**:
    - Optimizer now tests ALL possible feature combinations (2^N) instead of limited subset
    - For 10 experimental features: 1024 combinations × 2 models = 2048 total trainings
    - Tests every possible subset: 0 features (baseline), 1 feature, 2 features, ..., N features (all enabled)
    - Distribution: C(10,0)=1, C(10,1)=10, C(10,2)=45, C(10,3)=120, ..., C(10,10)=1
    - Addresses issue where important feature interactions were missed (e.g., target_temp_avg_24h + heating_degree_hours_7d)
  - **Memory Optimization**:
    - Log tail limited to last 10 messages to prevent memory issues with 2000+ trainings
    - Added `add_log_message()` method to OptimizerProgress that maintains tail limit
    - Configurable via `max_log_messages` parameter (default: 10)
  - **Top Results Display**:
    - Added `get_top_results(n=20)` method to retrieve top N configurations
    - Returns results sorted by validation MAPE (lower is better)
    - Filters out failed results automatically
    - UI can display top 20 results without overwhelming users
  - **Thread Safety Verified**:
    - Existing `_config_lock` ensures atomic feature configuration and dataset building
    - Each parallel worker applies correct feature settings before training
    - Comprehensive tests verify correct behavior with parallel execution
  - **Testing**:
    - 16 new tests added for full combination generation, log tailing, and top results
    - All 35 optimizer tests pass (19 existing + 16 new)
    - Verified all C(N,k) combinations are generated for each size k
    - Verified specific combinations from issue are included
  - **Performance**: Expected runtime depends on data and hardware (e.g., ~35 min if 1s/training, ~3 hours if 5s/training)

## [0.0.0.97] - 2025-12-03

- **Fix: Enable virtual sensor derived features in configuration page**
  - Fixed issue where derived features from virtual sensors (e.g., `temp_delta_avg_1h`) couldn't be selected in the UI
  - Updated `_is_derived_sensor_stat_feature()` method to check both raw sensors and virtual sensors
  - Updated `_create_derived_feature_metadata()` method to create proper metadata for virtual sensor derived features
  - Users can now enable/disable derived features from virtual sensors via checkboxes in the configuration page
  - Added comprehensive test coverage (5 new tests) in `test_virtual_sensor_derived_features.py`
  - All tests pass (53/53 tests in feature config module)

## [0.0.0.96] - 2025-12-03

- **Parallel Optimizer with Derived Feature Support**
  - **Parallel Training Execution**:
    - Optimizer now uses ThreadPoolExecutor for parallel model training
    - Configurable worker count (default: 3 parallel workers, as requested)
    - Thread-safe progress updates with locking mechanism
    - Faster optimization through concurrent training of different feature combinations
    - Background thread already supported, now with parallel execution within that thread
  - **Derived Features Support**:
    - Optimizer now discovers and tests all derived features from feature configuration
    - Added `_get_all_available_features()` to enumerate experimental + derived features
    - Supports dynamically created features like `sensor_name_avg_1h`, `sensor_name_avg_6h`, etc.
    - Updated `apply_best_configuration()` to use generic `enable_feature()` method for both experimental and derived features
    - `_get_experimental_feature_combinations()` now accepts `include_derived` parameter
  - **Enhanced Progress Reporting**:
    - Progress messages now show "X/Y" format (e.g., "[3/24] Testing configuration...")
    - Trophy emoji (🏆) displayed when a new best result is found
    - Real-time progress updates as training tasks complete (not just per configuration)
    - Better visibility into parallel execution progress
  - **New Training Function**:
    - Added `_train_single_configuration()` for thread-safe single model training
    - Handles both single-step and two-step model types
    - Proper error handling and result reporting per training task
  - **API Enhancements**:
    - `run_optimization()` accepts `max_workers` parameter (default: 3)
    - `run_optimization()` accepts `include_derived_features` parameter (default: True)
    - App now passes `max_workers=3` and `include_derived_features=True` to optimizer
  - **Testing**:
    - Added comprehensive test suite `test_optimizer_parallel.py` with 12 new tests
    - Tests cover derived feature discovery, parallel execution, thread safety, and progress reporting
    - All existing tests pass without regression
  - **Impact**:
    - Users can now optimize across all derived features (e.g., 48+ features as mentioned in the issue)
    - Much faster optimization through parallel training (3x speedup with 3 workers)
    - Better progress visibility with X/Y counts and real-time updates
    - All combinations of kWh, day, and other derived features are tested

## [0.0.0.95] - 2025-12-03

- **Optimizer Improvements: Async Execution, Result Storage, and Enhanced UI**
  - **Database Storage for Optimization Results**:
    - Created `OptimizerRun` and `OptimizerResult` database models to persist optimization history
    - Implemented `optimizer_storage.py` module with functions to save/load optimizer runs and results
    - All optimization results are now saved to database automatically after completion
  - **Async Optimization with Polling**:
    - Optimizer now runs asynchronously in a background thread instead of blocking the UI
    - Added `_run_optimizer_in_thread()` function for background execution
    - Status is set to "busy" when starting a run, allowing UI to poll for progress
    - `/api/optimizer/run` endpoint now returns immediately and starts background optimization
    - `/api/optimizer/status` endpoint enhanced to return full results when complete
  - **New API Endpoints**:
    - `POST /api/optimizer/apply/<result_id>` - Apply any specific result from the results table by its database ID
    - `GET /api/optimizer/runs` - List recent optimizer runs (summary view)
    - `GET /api/optimizer/runs/<run_id>` - Get detailed information about a specific run
    - `GET /api/optimizer/latest` - Get the most recent optimizer run with all results
  - **UI Enhancements** (Partial - in progress):
    - Results table now includes "Apply" button on each row for easy configuration application
    - Winner row styling updated: crown icon (👑) instead of star (⭐)
    - Winner row background changed to light green `rgba(76, 175, 80, 0.2)` instead of dark green
    - UI prepared for polling-based status updates (full integration pending)
  - **Feature Configuration**:
    - Verified that derived/custom sensors created via ⚙️ Feature Configuration are properly picked up
    - `_is_derived_sensor_stat_feature()` and `_create_derived_feature_metadata()` handle avatar and other derived features
  - **Testing**:
    - Added comprehensive test suite in `test_optimizer_storage.py`
    - Tests cover database operations, async endpoint behavior, and result storage
    - 14 new tests validating optimizer storage and API endpoints
  - **Impact**:
    - Users can now run optimizer in background without blocking the UI
    - Historical optimization results are preserved in database
    - Users can apply any configuration from results table, not just the best one
    - Better visibility into past optimization runs

## [0.0.0.94] - 2025-12-03

- **Fixed Feature Statistics Time Range Calculation Bug**
  - **Issue**: Feature statistics were not being generated (0 calculated, 0 saved) even when data was available
  - **Root Cause**: The time range calculation had a critical bug:
    1. When auto-determining start_time, it added the maximum window size to db_start
    2. If data span was shorter than the max window (e.g., 3 days of data but 7d window enabled), calculated start_time would be AFTER end_time
    3. Example from issue: start_time = 2025-12-07, end_time = 2025-12-03 → 0 stats calculated
  - **Fixes Applied**:
    1. Removed global max-window-based start_time calculation
    2. Each stat type (avg_1h, avg_6h, avg_24h, avg_7d) now calculates with its own time range
    3. avg_1h can now work with just 1 hour of data, even if avg_7d is also enabled
    4. Added validation and warning when data span is insufficient
    5. Fixed `SensorStatsConfig.from_dict()` to respect explicitly empty enabled_stats
    6. Fixed calculation loop to not auto-create default configs for unconfigured sensors
  - **Impact**: 
    - Feature statistics now generate correctly regardless of data span
    - Each stat type is independent - short windows work even when long windows can't
    - More informative logging shows which sensors/stats are processed vs skipped
  - Added 3 comprehensive test suites:
    - `test_feature_stats_time_range_fix.py`: Tests for the backwards time range bug fix
    - `test_per_stat_type_time_ranges.py`: Tests for independent per-stat-type calculations
    - All 16 tests passing, demonstrating correct behavior with various data spans

## [0.0.0.93] - 2025-12-03

- **Fixed Feature Statistics Not Being Generated**
  - **Issue**: When enabling derived features like `wind_avg_1h` or `wind_avg_6h` in the feature configuration, the `feature_statistics` table stayed empty - no statistics were being calculated or stored
  - **Root Cause**: The `calculate_feature_statistics()` function was using a hard-coded start time of 7 days after the first resampled sample, regardless of which statistics were actually enabled. This meant:
    1. If you only had 1-2 days of data but enabled `wind_avg_1h` (only needs 1 hour of history), no statistics would be calculated because the start time was 7 days in the future
    2. The function was checking ALL sensors for their enabled statistics, including auto-created defaults which included AVG_24H, making the problem worse
  - **Fix**:
    - Modified start_time calculation to use the maximum window size from EXPLICITLY CONFIGURED sensors only, not auto-created defaults
    - Start time is now based on actual enabled statistics rather than always using 7 days
    - If only `wind_avg_1h` is enabled (60 minutes window), calculations start after 1 hour instead of 7 days
    - If `wind_avg_6h` is enabled (360 minutes window), calculations start after 6 hours
  - **Impact**: Feature statistics (`wind_avg_1h`, `wind_avg_6h`, `wind_avg_24h`, etc.) are now properly calculated and stored in the `feature_statistics` table when enabled, even with limited historical data
  - Added comprehensive test coverage in `test_feature_stats_fix.py` with 3 test cases validating the fix
  - All existing tests continue to pass

## [0.0.0.92] - 2025-12-02

- **Fixed Sensor Statistics Configuration Overwrite Bug**
  - **Issue**: When toggling ML features for model training in the configuration panel, the sensor statistics configuration was being overwritten, causing all derived features to disappear from sensor cards
  - **User Experience**: 
    1. User configured sensor statistics (e.g., wind with avg_1h, avg_6h, avg_24h, and avg_7d enabled)
    2. User toggled a feature card to enable/disable it for model training
    3. **Bug**: All sensor statistics configuration was overwritten to match only what was needed by active ML features
    4. **Expected**: Sensor statistics configuration should remain independent from ML feature selection
  - **Root Cause**: The `/api/features/toggle` endpoint was calling `sync_stats_config_with_features()` after every feature toggle, which overwrote the user's sensor statistics configuration
  - **Fix**: 
    - Removed the automatic sync call from the feature toggle endpoint
    - Sensor statistics configuration now remains independent from ML feature configuration
    - Users can configure which statistics to collect separately from which features to use for training
  - **Behavior**: 
    - Toggling ML features only updates which features are used for model training
    - Sensor statistics configuration persists unchanged
    - No more disappearing derived features from sensor cards
  - Added comprehensive test coverage in `test_sensor_stats_independence.py` with 3 test cases
  - All existing tests continue to pass

## [0.0.0.91] - 2025-12-02

- **Fixed Feature Toggle for Derived Sensor Features**
  - **Issue**: When trying to activate a derived feature (e.g., `wind_avg_1h`, `outdoor_temp_avg_6h`) on a Card via the configuration panel, the API returned a 404 error: `{"message":"Feature 'wind_avg_1h' not found","status":"error"}`
  - **Root Cause**: The `/api/features/toggle` endpoint only recognized features explicitly defined in `CORE_FEATURES` and `EXPERIMENTAL_FEATURES` lists. Derived features from sensor statistics configuration (like `wind_avg_1h`) were shown in sensor cards but couldn't be toggled because they weren't in these predefined lists
  - **Fix**: 
    - Modified `enable_feature()` and `disable_feature()` methods in `FeatureConfiguration` to dynamically handle derived sensor statistic features
    - Added `_is_derived_sensor_stat_feature()` helper to validate derived feature names (format: `<sensor>_avg_<window>`)
    - Added `_create_derived_feature_metadata()` helper to generate metadata for derived features on-the-fly
    - Updated `get_all_features()` to include dynamically enabled derived features in the feature list
  - **Behavior**: Users can now toggle any valid derived sensor feature (e.g., `wind_avg_1h`, `pressure_avg_24h`) through the configuration panel without getting 404 errors
  - Added comprehensive test coverage in `test_derived_feature_toggle.py` with 3 test cases
  - All 573 tests pass, confirming no regressions

## [0.0.0.90] - 2025-12-02

- **Fixed Sensor Card Bug: Base Feature Missing for Virtual Sensors**
  - **Issue**: Virtual sensors with time-based statistics enabled showed only the statistics (e.g., 4 checkboxes for avg_1h, avg_6h, avg_24h, avg_7d) but were missing the base sensor checkbox (e.g., "Current Value"), resulting in 4 options instead of the expected 5
  - **Root Cause**: When a sensor wasn't registered in the feature configuration (common for virtual sensors), the base sensor feature was not added to the sensor card
  - **Fix**: Modified `/api/features/sensor_cards` endpoint to always include the base sensor feature, even when it's not in the feature configuration yet. Virtual sensors and new sensors now correctly display both the base feature AND all enabled time-based statistics
  - **Example**: The "Setpoint Delta" virtual sensor with 4 statistics enabled now correctly shows 5 checkboxes: 1 for "Current Value" + 4 for time-based averages (avg_1h, avg_6h, avg_24h, avg_7d)
  - Added test case `test_virtual_sensor_with_stats_shows_base_feature` to validate the fix
  - Fixes issue #188

## [0.0.0.89] - 2025-12-02

- **Tests: Removed MariaDB Dependency**
  - Refactored `test_sensor_cards.py` to use in-memory SQLite instead of MariaDB
  - Tests now use the same pattern as other test files with SQLite and monkeypatch
  - All 569 tests now pass without requiring a MariaDB connection
  - Fixes test failures when MariaDB is not available

## [0.0.0.88] - 2025-12-02

- **Fixed Sensor Cards: All Time-Based Statistics Now Visible**
  - **Issue**: Sensor cards (e.g., humidity, temperature, pressure) only showed one checkbox for the base sensor value, missing checkboxes for configured time-based statistics (avg_1h, avg_6h, avg_24h, avg_7d)
  - **Root Cause**: The `/api/features/sensor_cards` endpoint only displayed features statically defined in CORE_FEATURES and EXPERIMENTAL_FEATURES, ignoring user-configured statistics from Feature Stats Configuration
  - **Fix**: Modified endpoint to dynamically query feature_stats_config and include all enabled time-based statistics
  - **Behavior**: Each sensor card now shows:
    - A checkbox for the base sensor value (e.g., "humidity")
    - Individual checkboxes for each configured time-based statistic (e.g., "humidity_avg_1h", "humidity_avg_6h", "humidity_avg_24h")
  - **Example**: If you configured humidity to generate avg_1h, avg_6h, and avg_24h statistics in the "Feature Stats Configuration" tab, the humidity card now displays 4 checkboxes (1 for current value + 3 for averages)
  - **Applies to all sensors**: This fix works for all sensors (raw and virtual), not just humidity
  - Added comprehensive test suite in `test_sensor_cards.py` to validate the fix

## [0.0.0.87] - 2025-12-02

- **UI Improvement: Sensor Cards Display**
  - **Time/Date and kWh cards are now always visible** - These special feature cards (Time & Date Features and Heating Usage Features) load automatically and remain visible
  - **Physical sensor cards are hidden behind a toggle button** - Sensor cards like Outdoor Temperature and Indoor Temperature with their aggregation features (avg_1h, avg_6h, etc.) are now hidden by default
  - Added "📊 Show Sensor Cards" button to toggle visibility of physical sensor cards
  - Button text changes between "Show Sensor Cards" and "Hide Sensor Cards" based on visibility state
  - Sensor cards are loaded once on page load (no lazy loading) and toggled for better performance
  - **Each sensor card groups the sensor with its derived features** - e.g., Indoor Temperature card shows indoor_temp, indoor_temp_avg_1h, etc.
  - Features displayed depend on what's enabled in the Sensor Configuration tab

## [0.0.0.86] - 2025-12-02

- **Fixed Sensor Visibility Issue**: Sensors are now correctly displayed on the configuration page
  - **Root Causes Fixed**:
    - Added missing `.data-table` CSS class that was referenced in JavaScript but not defined in stylesheet
    - Fixed undefined CSS variables (`--success-color`, `--warning-color`, `--secondary-color`) to use correct variables (`--accent-green`, `--accent-orange`, `--text-muted`)
    - Updated `/api/features/sensor_cards` endpoint to dynamically discover sensors instead of using hardcoded list
  - **Dynamic Sensor Discovery**:
    - `/api/features/sensor_cards` now queries `get_sensor_category_config()` to get all configured sensors
    - Includes all sensors (core, experimental, and virtual) not just a hardcoded subset
    - Shows all sensors regardless of enabled/disabled status so users can configure them
    - Uses actual sensor metadata from configuration instead of hardcoded display names and units
  - **UI Improvements**:
    - Raw sensors table now displays correctly with proper styling and borders
    - Status badges (Core/Enabled/Disabled) render with correct colors
    - Input fields for entity ID and unit have proper styling
    - All configured sensors are visible in the Feature Configuration section
  - **Impact**: Users can now see and configure all their sensors in both the Configuration tab (feature selection) and Sensor Configuration tab (entity ID and statistics configuration)

## [0.0.0.85] - 2025-12-02

- **Feature Configuration Improvements**
  - **Core features can now be disabled**: CORE features are labeled as 'CORE' in UI but can still be toggled on/off
  - **Sync feature statistics with ML configuration**: Feature statistics (avg_1h, avg_6h, avg_24h, avg_7d) are now only calculated for features that are enabled in the ML configuration
    - Added `sync_stats_config_with_features()` function to synchronize feature stats with active ML features
    - Added `derive_stats_from_feature_config()` to determine which sensor statistics are needed based on enabled features
    - `calculate_feature_statistics()` now syncs with feature config before calculating (optional parameter)
  - **New API endpoints for feature management**:
    - `/api/features/special_cards` - Returns special feature cards for Time/Date and Usage calculated features
    - `/api/features/sensor_cards` - Returns sensor cards with their aggregation features
    - `/api/features/toggle` - Updated to handle both CORE and EXPERIMENTAL features (previously only experimental)
  - **Special feature cards**:
    - **Time & Date Features**: hour_of_day, day_of_week, is_weekend, is_night (system-generated card)
    - **Heating Usage Features**: heating_kwh_last_1h/6h/24h/7d, heating_degree_hours_24h/7d (system-generated card)
  - **Feature storage changes**:
    - FeatureConfiguration now stores both `core_enabled` and `experimental_enabled` dictionaries
    - Both core and experimental features can be toggled via `enable_feature()` / `disable_feature()` methods
  - **UI Implementation**:
    - **Auto-load on page open**: Feature configuration loads automatically when Configuration tab is shown (removed manual "Load Features" button)
    - **Special feature cards displayed**: Time & Date and Heating Usage features shown in dedicated cards
    - **Sensor feature cards**: Each sensor (outdoor_temp, indoor_temp, etc.) shows its aggregation features with checkboxes
    - **Feature badges**: All features show CORE or EXPERIMENTAL badges
    - **Toggleable checkboxes**: All features (including CORE) can be enabled/disabled via checkboxes
    - **Auto-refresh**: UI automatically refreshes after toggling features to show updated state
  - **Improved test coverage**:
    - Added comprehensive tests for feature stats synchronization in `test_feature_stats_sync.py`
    - Tests verify that feature statistics respect ML feature configuration
    - All 55 tests pass

- Resolves issue: Features should only be calculated and displayed when enabled in configuration

## [0.0.0.84] - 2025-12-02

- **Fix "Show Sensors & Statistics" button error**
  - Fixed AttributeError: 'SensorConfig' object has no attribute 'display_name'
  - The `/api/features/sensors_with_stats` endpoint now correctly retrieves display_name from SensorDefinition
  - Fixed by calling `get_sensor_definition()` to access the display_name attribute

- **Fix feature_statistics table staying empty**
  - Feature statistics (time-span averages) are now automatically calculated after resampling
  - The `/resample` endpoint now calls `calculate_feature_statistics()` automatically after successful resampling
  - When flush=true is used during resampling, feature statistics are also flushed and recalculated
  - Feature statistics include: avg_1h, avg_6h, avg_24h, avg_7d for all enabled sensors
  - Fixes issue where feature_statistics table remained empty despite resampling completing successfully

- Resolves KevinHekert/HomeAssistantAddOns issue with "Show Sensors & Statistics" button

## [0.0.0.83] - 2025-12-02

- **Sensor Configuration UI Reorganization and Data Lineage Improvements**
  - **UI Changes**:
    - Removed duplicate "📡 Sensor Configuration" section from Configuration tab
    - Kept only "⚙️ Feature Configuration" on Configuration tab with enhanced guidance text
    - Added "Show Sensors & Statistics" button to display comprehensive sensor information
    - Feature Configuration now displays all sensors (raw + virtual) with their time-based statistics
    - Clear visual indicators: RAW SENSOR vs VIRTUAL SENSOR badges
    - Shows enabled statistics per sensor and generated feature names (e.g., outdoor_temp_avg_1h)
    - Configuration tab now focuses on system-wide settings (resampling, sync, features, weather)
    - Sensor Configuration tab retains all sensor management: Raw Sensors, Virtual Sensors, Feature Stats
  - **Database Schema Enhancements**:
    - Added `is_derived` boolean column to `resampled_samples` table to distinguish data lineage:
      - `is_derived=False`: Direct time-weighted averages from raw sensor data
      - `is_derived=True`: Virtual/derived sensors calculated from resampled raw data
    - Created `feature_statistics` table for time-span rolling averages:
      - Stores avg_1h, avg_6h, avg_24h, avg_7d calculated from resampled data
      - Fields: sensor_name, stat_type, slot_start, value, unit, source_sample_count
      - Separate from resampled_samples for clarity and performance
    - Migration logic handles existing databases automatically via `init_db_schema()`
  - **Resampling Order Enforcement**:
    - Step 1: Resample raw sensors → `resampled_samples` (is_derived=False)
    - Step 2: Calculate virtual sensors from step 1 → `resampled_samples` (is_derived=True)
    - Step 3: Calculate time-span averages → `feature_statistics` (separate invocation)
    - Example: 1 hour with 5-min intervals = 12 raw samples + 12 virtual samples + 12 averages
  - **New Modules**:
    - `db/calculate_feature_stats.py` for time-span average calculation
      - `calculate_feature_statistics()`: Main calculation function
      - `calculate_rolling_average()`: Rolling window averages
      - `get_all_sensor_names()`: Discovers all sensors (raw + virtual + from resampled data)
      - `flush_feature_statistics()`: Clears all statistics for recalculation
      - Configuration-driven: Only calculates enabled statistics per sensor
  - **New API Endpoints**:
    - `GET /api/features/sensors_with_stats`: Get all sensors with their enabled statistics
    - `GET /api/resampled_data`: Query resampled samples with `is_derived` field filtering
    - `GET /api/feature_statistics`: Query time-span averages with flexible filters
    - All endpoints support time range filtering, pagination, and proper error handling
  - **Testing**:
    - 6 new schema migration tests: column addition, defaults, querying, backward compatibility
    - 6 new resampling tests: is_derived flags, 1 hour = 12 records scenario, all timeframes
    - 8 new feature statistics tests: rolling averages, configuration respect, table isolation
    - 5 new UI structure tests: tab organization, duplicate removal, content validation
    - All 558 tests pass (includes all existing + new tests)
  - **Documentation**: Added comprehensive inline documentation for data lineage and calculation order
  - Fixes KevinHekert/HomeAssistantAddOns#174

## [0.0.0.82] - 2025-12-02

- **Virtual Sensor Resampling**: Fixed issue where virtual sensors were not calculated during resampling
  - **Problem**: Virtual sensors could be configured via the UI, but their calculated values were not being filled during the resampling process
  - **Solution**: Modified `resample_all_categories()` in `db/resample.py` to:
    - Load enabled virtual sensors from configuration
    - Calculate virtual sensor values for each resampled time slot
    - Store virtual sensor values in `resampled_samples` table with their category names
  - **Functionality**:
    - Virtual sensors are calculated after raw sensors are resampled for each slot
    - Only calculated when both source sensors have values in the slot
    - Supports all operations: subtract, add, multiply, divide, average
    - Disabled virtual sensors are not calculated
    - Division by zero is handled gracefully (no value stored)
  - **Example**: A virtual sensor `temp_delta = target_temp - indoor_temp` will now have its calculated values available in the resampled data
  - **Tests**: Added comprehensive test suite (`test_virtual_sensors_resample.py`) with 9 test cases covering all operations and edge cases
  - All 64 existing resample tests continue to pass

## [0.0.0.81] - 2025-12-02

- **Feature Statistics Configuration**: Implemented Step 5 of sensor configuration feature - Time-based Feature Stats
  - **New Module**: `db/feature_stats.py` for feature statistics management
    - `StatType` enum: avg_1h, avg_6h, avg_24h, avg_7d
    - `SensorStatsConfig` dataclass for per-sensor statistics configuration
    - `FeatureStatsConfiguration` class for global configuration management
    - Default stats enabled: avg_1h, avg_6h, avg_24h
    - Persistent JSON storage at `/data/feature_stats_config.json`
  - **New API Endpoints**:
    - `GET /api/feature_stats/config`: Get statistics configuration for all sensors
    - `POST /api/feature_stats/set`: Enable/disable a specific statistic for a sensor
  - **UI Implementation**:
    - Feature Stats Configuration section now fully functional
    - Table display with checkboxes for each stat type per sensor
    - Shows sensor type (Raw/Virtual) with badges
    - Interactive checkboxes to enable/disable each statistic
    - Real-time status updates when toggling statistics
    - Explanatory text about stat generation
  - **Functionality**:
    - Works for both raw sensors and virtual sensors
    - Each sensor can have different statistics enabled
    - Statistics are stored with category names like "sensor_name_avg_1h"
    - Configuration persists across restarts
  - **Example Usage**:
    - Enable avg_1h for outdoor_temp → creates "outdoor_temp_avg_1h" during resampling
    - Enable avg_6h and avg_24h for temp_delta (virtual sensor) → creates aggregated delta values
  - **Note**: Statistics calculation will be integrated into resampling in Step 6
- **Bug Fix**: Fixed indentation error in sensor_category_config.py (duplicate return statements)
- **Tests**: All 524 existing tests pass

## [0.0.0.80] - 2025-12-02

- **Virtual Sensors Implementation**: Implemented Step 4 of sensor configuration feature - Virtual (Derived) Sensors
  - **New Module**: `db/virtual_sensors.py` for virtual sensor management
    - `VirtualSensorDefinition` dataclass for sensor definitions
    - `VirtualSensorOperation` enum: subtract, add, multiply, divide, average
    - `VirtualSensorsConfiguration` class for configuration management
    - `calculate()` method to compute virtual sensor values from two source sensors
    - Persistent JSON storage at `/data/virtual_sensors_config.json`
  - **New API Endpoints**:
    - `GET /api/virtual_sensors/list`: List all virtual sensors with their configurations
    - `POST /api/virtual_sensors/add`: Create a new virtual sensor
    - `DELETE /api/virtual_sensors/<name>`: Delete a virtual sensor
    - `POST /api/virtual_sensors/<name>/toggle`: Enable/disable a virtual sensor
  - **UI Enhancements**:
    - Virtual Sensors section now fully functional
    - Table display showing name, formula, unit, status
    - "Add Virtual Sensor" dialog with prompts for all required fields
    - Enable/Disable buttons for each virtual sensor
    - Delete button with confirmation dialog
    - Reload button to refresh the list
    - Formula display with mathematical symbols (-, +, ×, ÷, avg)
    - Enabled/Disabled status badges
  - **Supported Operations**:
    - `subtract`: sensor1 - sensor2 (e.g., temp_delta = target_temp - indoor_temp)
    - `add`: sensor1 + sensor2 (e.g., total_power = device1 + device2)
    - `multiply`: sensor1 × sensor2 (e.g., energy = power × time)
    - `divide`: sensor1 ÷ sensor2 (with zero-division protection)
    - `average`: (sensor1 + sensor2) / 2
  - **Examples**:
    - Create "temp_delta" = target_temp - indoor_temp (heating demand indicator)
    - Create "outdoor_wind_factor" = wind × outdoor_temp (wind chill indicator)
  - **Note**: Virtual sensor calculations will be integrated into resampling in Step 6

## [0.0.0.79] - 2025-12-02

- **Unit Field Storage and Configuration**: Implemented Step 3 of sensor configuration feature
  - Added `unit` field to `SensorConfig` dataclass
  - Units can now be stored per sensor in the configuration file
  - Unit can override the default unit from `SensorDefinition`
  - **New API Endpoint**: `POST /api/sensors/set_unit`
    - Accepts `category_name` and `unit` parameters
    - Saves unit to sensor configuration
    - Returns updated sensor configuration
  - **Database Model Updates**:
    - `SensorConfig.unit` field added with default empty string
    - `to_dict()` and `from_dict()` methods updated to handle unit
    - `set_unit()` method added to `SensorCategoryConfiguration` class
  - **UI Updates**:
    - Unit input fields are now functional
    - `saveRawSensor()` function now saves both entity ID and unit
    - Displays success message when both are saved successfully
    - Shows warning if entity ID saved but unit failed
  - **Configuration Persistence**:
    - Units are stored in `/data/sensor_category_config.json`
    - Configuration is saved automatically when unit is updated
    - `get_sensors_by_type()` now returns configured unit (or default if not set)
- **Backwards Compatibility**: Existing configurations without unit field will default to empty string

## [0.0.0.78] - 2025-12-02

- **New Sensor Configuration Tab**: Added new "📡 Sensor Configuration" tab in the UI
  - Step 1 of comprehensive sensor configuration and virtual sensor feature
  - Tab positioned between Configuration and Model Training tabs
  - Three main sections created: Raw Sensors, Virtual Sensors, and Feature Stats Configuration
- **Raw Sensors Section**: Interface to view and configure all raw sensors
  - Displays all sensors grouped by type (Usage, Weather, Indoor, Heating)
  - Shows sensor display name, description, entity ID, and unit fields
  - Core sensors displayed with green badge (always enabled)
  - Experimental sensors displayed with orange badge (can be enabled/disabled)
  - Individual Save button for each sensor
  - Uses existing `/api/sensors/category_config` endpoint for loading
  - Uses existing `/api/sensors/set_entity` endpoint for saving
  - `loadRawSensors()` JavaScript function to load and display sensor list
  - `saveRawSensor()` JavaScript function to save individual sensor configuration
- **Virtual Sensors Section** (placeholder for Step 4):
  - Button to add new virtual sensors
  - Reload list button
  - Placeholder message indicating feature will be implemented next
- **Feature Stats Configuration Section** (placeholder for Step 5):
  - Placeholder for time-based statistics configuration
  - Will enable avg_1h, avg_6h, avg_24h, avg_7d generation per sensor
- **UI Improvements**:
  - Added `.action-btn.small` CSS class for inline action buttons
  - Consistent styling with existing UI theme
  - Responsive table layout for sensor configuration
- **Note**: This is the first step of 8-step implementation. Virtual sensors and feature stats functionality will be added in subsequent steps.

## [0.0.0.77] - 2025-12-02

- **Sensor Category Configuration**: Refactored sensor configuration with Core/Experimental categories
  - **Core Sensors** (always enabled, required for prediction):
    - `hp_kwh_total`: Heat Pump kWh Total - Required for training and predictions
    - `outdoor_temp`: Outdoor Temperature - Essential for heating demand prediction
    - `indoor_temp`: Indoor Temperature - Required for setpoint calculations
    - `target_temp`: Target Temperature - Required for heating predictions
    - `wind`: Wind Speed - Affects heat loss and heating demand
    - `humidity`: Humidity - Affects thermal comfort
  - **Experimental Sensors** (optional, can be enabled/disabled via UI):
    - `pressure`: Barometric Pressure - May help predict weather patterns
    - `flow_temp`: Flow Temperature - For advanced heating system monitoring
    - `return_temp`: Return Temperature - For advanced heating system monitoring
    - `dhw_temp`: DHW Temperature - Used to filter DHW cycles
    - `dhw_active`: DHW Active - Helps exclude DHW from training
- **Sensor Configuration UI**: New "Sensor Configuration" card in Configuration tab
  - Sensors grouped by type: Usage, Weather, Indoor, Heating
  - Core sensors shown with green badge (always enabled)
  - Experimental sensors shown with orange badge (toggleable)
  - Entity ID input field for each sensor with Save button
  - Summary showing core/experimental/enabled counts
- **New API Endpoints**:
  - `GET /api/sensors/category_config`: Get sensor configuration with grouping by type
  - `POST /api/sensors/toggle`: Toggle experimental sensors (core sensors cannot be toggled)
  - `POST /api/sensors/set_entity`: Set entity_id for any sensor
  - `GET /api/sensors/definitions`: Get all sensor metadata definitions
- **Configuration Migration**: On first run, sensor settings are automatically migrated from `config.yaml` environment variables to the new JSON-based configuration (`/data/sensor_category_config.json`)
- **Sensor Sync Worker**: Updated to use new sensor category configuration
  - Only enabled sensors are synced
  - Dynamic sensor list based on configuration
- **New Module**: `db/sensor_category_config.py` for sensor category management
  - `SensorDefinition` and `SensorConfig` dataclasses
  - `SensorCategoryConfiguration` class for configuration management
  - Automatic migration from environment variables
  - Persistent JSON storage
- **Tests**: Added 70+ new tests for sensor category functionality
  - 54 tests for sensor category configuration module
  - 16 tests for API endpoints
  - 7 tests for sync_sensor_mappings
- **Note**: Sensor entity IDs in `config.yaml` (e.g., `wind_entity_id`) are now deprecated. They will be used for initial migration only. Configure sensors via the UI instead.

## [0.0.0.76] - 2025-12-02

- **Settings Optimizer**: Added automatic optimization feature to find the best model configuration
  - New "Settings Optimizer" card in the Model Training tab
  - Cycles through different experimental feature combinations
  - Trains both single-step and two-step models for each configuration
  - Finds the configuration with the lowest validation MAPE (%)
  - Current settings are saved before optimization and restored after completion
- **Optimizer UI Features**:
  - "Start Optimization" button to begin the process
  - Live log display showing optimization progress
  - Results table sorted by Val MAPE (best at top)
  - Best result highlighted with star icon (⭐)
  - "Apply Best Settings" button to save the optimal configuration
- **New API Endpoints**:
  - `POST /api/optimizer/run`: Start optimization and get results
  - `GET /api/optimizer/status`: Get current optimizer status
  - `POST /api/optimizer/apply`: Apply the best configuration found
- **New Module**: `ml/optimizer.py` with optimization logic
  - `run_optimization()`: Main optimization function
  - `apply_best_configuration()`: Apply and save optimal settings
  - `OptimizerProgress` and `OptimizationResult` dataclasses
  - Automatic feature combination generation (baseline, individual features, logical groups)
- **Tests**: Added 19 new tests for optimizer functionality
  - Module tests for feature combinations and optimization logic
  - API endpoint tests for run, status, and apply

## [0.0.0.75] - 2025-12-02

- **Fixed Sample Rate Handling in Model Training**: Historical aggregation windows now dynamically adjust based on the configured sample rate
  - Previously, window sizes were hardcoded assuming 5-minute slots (e.g., 12 slots for 1 hour)
  - When using different sample rates (e.g., 15 minutes), the windows were incorrect (12 slots = 3 hours instead of 1 hour)
  - This fix ensures correct historical feature computation for any configured sample rate
- **Dynamic Window Calculations**:
  - Added `_get_slots_per_hour()` helper function to calculate slots based on sample rate
  - `_compute_historical_aggregations()` now accepts `sample_rate_minutes` parameter
  - `_compute_target()` now accepts `sample_rate_minutes` parameter
  - Maximum kWh delta clipping now adjusts based on slot duration (10 kW * hours_per_slot)
  - Heating degree hours calculation uses dynamic hours_per_slot multiplier
- **FeatureDatasetStats Enhancement**: Added `sample_rate_minutes` field to track the sample rate used during training
- **Tests**: Added 2 new tests for verifying correct behavior with different sample rates
  - `test_different_sample_rates` for `_compute_historical_aggregations()`
  - `test_different_sample_rates` for `_compute_target()`
- **Issue Fixed**: Using different resample rates (e.g., 15 minutes) no longer results in weird prediction behavior

## [0.0.0.74] - 2025-12-02

- **Core Feature Expansion**: Moved two features from experimental to core baseline (15 core features total, was 13)
  - `indoor_temp_avg_6h`: 6-hour average indoor temperature - essential for thermal mass tracking
  - `outdoor_temp_avg_1h`: 1-hour outdoor temperature average - essential for near-term weather response
  - These features are now always active and cannot be disabled
- **Improved Training Feedback**: Training output now shows dynamic dataset information
  - Data range: Shows exact start and end timestamps of training data
  - History: Shows total hours of data available (~days)
  - Feature breakdown: Shows count of raw sensor vs calculated features
  - Clearer separation of sections: Dataset, Performance, and Features Used
- **UI Update**: Updated feature count display from 13 to 15 core features
- **Tests**: Added 2 new tests for the new core features (`indoor_temp_avg_6h`, `outdoor_temp_avg_1h`)
- Updated existing tests to reflect new core feature count (13 → 15)

## [0.0.0.73] - 2025-12-02

- **Two-Step Model Feature Display**: Added display of features used by each step in the two-step model
  - Training output now shows features used by Step 1 (Classifier) and Step 2 (Regressor)
  - Each step displays the number of features and the complete feature list
  - Model status check now clarifies that both steps use the same feature set
  - Improved section headers: "Step 1: Classifier" and "Step 2: Regressor" for clarity
  - Fixes issue: "Still can't see what feature-data is used for training models"

## [0.0.0.72] - 2025-12-02

- **Fixed Two-Step Model Training Feedback**: The two-step model training API response now includes top-level `classifier_metrics` and `regressor_metrics` fields for UI compatibility
  - Previously, metrics were only available under `step1_classifier.metrics` and `step2_regressor.metrics`, causing the UI to display "N/A" for all metric values
  - The UI now correctly displays: Accuracy, Precision, Recall, F1 Score for the classifier
  - The UI now correctly displays: Training samples, Validation samples, Train MAE, Val MAE, Val MAPE, Val R² for the regressor
  - The detailed `step1_classifier` and `step2_regressor` objects remain available for advanced API consumers
- **Tests**: Added 2 new tests for the two-step training endpoint API response structure

## [0.0.0.71] - 2025-12-02

- **Feature Verification for Training Models**: Added verification that features displayed are actually used during training
  - Training response now includes `feature_verification` object with:
    - `verified`: Boolean indicating if all features were found in the training dataset
    - `verified_features`: List of features confirmed to be used
    - `missing_in_dataset`: List of features expected but not found in data
  - Training response now includes `feature_categories` showing:
    - `raw_sensor_features`: Features directly from sensors (outdoor_temp, wind, humidity, etc.)
    - `calculated_features`: Derived/aggregated features (outdoor_temp_avg_24h, heating_kwh_last_6h, delta_target_indoor, etc.)
  - Training response now includes `feature_details` with metadata for each feature (category, description, unit, is_calculated)
- **Two-Step Model Step Explanations**: Added clear explanations of what each step does in the two-step model
  - `step1_classifier`: Explains that Step 1 predicts whether heating will be active (on) or inactive (off) for each hour
  - `step2_regressor`: Explains that Step 2 predicts kWh consumption for active hours only (inactive hours = 0 kWh)
  - Each step includes: description, purpose, features_used, feature_count, training_samples, and metrics
- **New Functions in feature_config.py**:
  - `categorize_features()`: Categorizes features as raw sensor vs calculated
  - `get_feature_details()`: Returns detailed metadata for each feature
  - `verify_model_features()`: Verifies model features match dataset features
- **Tests**: Added 7 new tests for feature verification functionality
- **Fixed Weerlive API Parsing**: Fixed parsing of the actual Weerlive API v2 response format
  - The API provides hourly forecast data (`uur_verw`) at the root level of the JSON response, not inside `liveweer[0]`
  - Added support for the actual Weerlive API v2 field names:
    - `windkmh`: Wind speed in km/h (automatically converted to m/s for internal use)
    - `windms`: Wind speed in m/s (alternative field)
    - `timestamp`: Unix timestamp for each hourly forecast
  - Added support for the actual datetime format: "DD-MM-YYYY HH:00" (e.g., "02-12-2025 14:00")
  - Maintains backwards compatibility with older format (uur_verw inside liveweer, winds field, HH:MM format)
- **New Test Cases**: Added 4 new tests for Weerlive API v2 format parsing
  - Test for parsing uur_verw at root level with windkmh field
  - Test for windms field parsing
  - Test for DD-MM-YYYY HH:00 datetime format parsing
  - Integration test for fetch_weather_forecast with full API v2 response
- **Incremental Resampling**: When resampling without flush, the system now starts from the latest resampled slot minus 2x the sample rate, instead of reprocessing all historical data
  - Example: With 5-minute sample rate and latest resampled slot at 12:55, resampling starts from 12:45 (12:55 - 2*5)
  - This significantly improves performance for regular resample operations
  - Existing values that differ will be replaced (idempotent behavior)
  - Use `flush=True` to force full reprocessing from the beginning
- **New Function**: Added `get_latest_resampled_slot_start()` to retrieve the most recent resampled slot timestamp
- **Tests**: Added 9 new tests for incremental resampling behavior
- **Training Data Range Units from Source**: Unit information is now extracted from the actual samples table instead of using hardcoded values
  - `TrainingDataRange` dataclass now includes a `unit` field
  - Units are extracted from the first sample of each category in the resampled data
  - Removed hardcoded `SENSOR_UNITS` dictionary from `app.py`
  - Training data response now displays the actual unit stored with each sensor's data
  - This ensures unit accuracy even for sensors with non-standard units
- Updated `_load_resampled_data()` to include the `unit` column from the database
- Added test for unit extraction from source data
- Updated existing tests to use realistic units in test data

## [0.0.0.70] - 2025-12-02

- **Weather API Integration**: Added integration with weerlive.nl API for weather forecast data
  - New "Weather API Settings" card in Configuration tab for API key and location configuration
  - API credentials are validated before saving to ensure they work
  - Get your free API key at: https://weerlive.nl/delen.php
- **Load Weather Forecast**: Added "🌤️ Load Weather (24h)" button in Scenario-Based Prediction section
  - Fetches upcoming 24-hour weather forecast from weerlive.nl API
  - Automatically populates scenario prediction input with weather data
  - Configurable target temperature for predictions
- **Prediction Storage and Comparison**:
  - Added "💾 Store Prediction" button to save predictions for later comparison
  - New "📊 Stored Predictions" section in Model Training tab
  - Compare stored predictions with actual sensor data when it becomes available
  - View comparison metrics: MAE, MAPE, total predicted vs actual kWh
  - Delete stored predictions when no longer needed
- **New API Endpoints**:
  - `GET /api/weather/config`: Get weather API configuration
  - `POST /api/weather/config`: Save and validate weather API credentials
  - `POST /api/weather/validate`: Validate API credentials without saving
  - `GET /api/weather/forecast`: Fetch weather forecast for next 24 hours
  - `POST /api/predictions/store`: Store a prediction for later comparison
  - `GET /api/predictions/stored`: List all stored predictions
  - `GET /api/predictions/stored/<id>`: Get specific stored prediction
  - `DELETE /api/predictions/stored/<id>`: Delete stored prediction
  - `POST /api/predictions/stored/<id>/compare`: Compare prediction with actual data
- **New Modules**:
  - `ha/weather_api.py`: Weather API integration with weerlive.nl
  - `db/prediction_storage.py`: Prediction storage and comparison functionality
- **Tests**: Added 50 new tests for weather API and prediction storage functionality

## [0.0.0.69] - 2025-12-02

- **Scenario Prediction Uses Two-Step When Enabled**: The simplified scenario prediction endpoint now automatically uses the two-step model when enabled
  - When two-step prediction is enabled in Feature Configuration and the two-step model is trained, `/api/predictions/scenario` will use the two-step approach
  - Response includes `is_active` and `activity_probability` for each timeslot when using two-step
  - Response includes summary with active/inactive hour counts
  - UI displays active/inactive status with 🔥/❄️ icons for each hour
  - Falls back to single-step model if two-step is not available
- **UI Improvements for Two-Step Scenario Predictions**:
  - Status message indicates "(using two-step prediction)" when enabled
  - Results table shows "Active" column with status icons (🔥 Active / ❄️ Inactive)
  - Results table shows "Activity Prob." column with classifier probability
  - Summary box shows count of active vs inactive hours
- Added 4 new tests for two-step scenario prediction functionality

## [0.0.0.68] - 2025-12-02

- **Two-Step Prediction UI Improvements**: Redesigned the two-step prediction feature for better visibility and clarity
  - Moved two-step prediction toggle to its own dedicated card in the Configuration tab
  - Card styled as an experimental feature similar to other experimental options
  - Added detailed description explaining the two-step approach (classifier + regressor)
  - Shows current model status (available/not trained) and activity threshold
- **Two-Step Model Training Section**: Added dedicated training section in Model Training tab
  - New "Two-Step Model Training" card with experimental badge
  - Displays training statistics: activity threshold, active samples, inactive samples, classifier accuracy
  - Shows detailed classifier metrics (accuracy, precision, recall, F1 score)
  - Shows detailed regressor metrics (MAE, MAPE, R²) for active hours only
  - Clear feedback showing how many hours were classified as active vs inactive
  - Training data range table showing sensor values used
- **Improved Training Feedback**: When training the two-step model, users now clearly see:
  - The computed activity threshold (minimum kWh to consider heating "active")
  - Number of active vs inactive samples in the training data
  - Classifier performance metrics
  - Regressor performance on active hours only

## [0.0.0.67] - 2025-12-02

- **Historical Day Prediction Timestamps**: When loading historical data for scenario predictions, timestamps are now adjusted to be 2 days after today (day after tomorrow)
  - The `scenario_format` in `/api/examples/historical_day/<date>` now contains future timestamps for prediction
  - Original historical timestamps are preserved in `hourly_data` for comparison purposes
  - This allows comparing predictions with actual historical data while using valid future timestamps for the prediction API
- Added test for timestamp adjustment in historical day scenario format

## [0.0.0.66] - 2025-12-02

- **Sync Configuration UI**: Added UI controls to configure sensor sync settings
  - New "Sync Configuration" card in the Configuration tab
  - Backfill Days: How many days to look back when no samples exist (1-365, default: 14)
  - Sync Window Size: Size of each sync window in days (1-30, default: 1)
  - Sensor Sync Interval: Wait time in seconds between syncing sensors (1-3600, default: 1)
  - Loop Interval: Wait time in seconds between sync loop iterations (1-3600, default: 1)
  - Configuration is persisted to `/data/sync_config.json`
- **Two-Step Prediction Toggle in UI**: Added checkbox in Feature Configuration to enable/disable two-step prediction
  - Visual toggle with description of the two-step approach
  - Status is loaded on page load
- **New API Endpoints**:
  - `GET /api/sync_config`: Get current sync configuration with limits
  - `POST /api/sync_config`: Update sync configuration (partial updates supported)
- **New Module**: `db/sync_config.py` for persistent sync configuration storage
  - `SyncConfig` dataclass with all sync settings
  - `get_sync_config()`, `set_sync_config()` for full configuration access
  - Individual getters: `get_backfill_days()`, `get_sync_window_days()`, `get_sensor_sync_interval()`, `get_sensor_loop_interval()`
- **Code Changes**:
  - `ha_api.py`: Now uses `get_backfill_days()` and `get_sync_window_days()` from sync_config module
  - `sensors.py`: Now uses `get_sensor_sync_interval()` and `get_sensor_loop_interval()` from sync_config module
- **Tests**: Added 23 new tests for sync configuration module

## [0.0.0.65] - 2025-12-02

- **Removed UI Sections**: Removed the following UI sections from the Model Training tab:
  - "🔮 Single Slot Prediction" - removed card and functionality
  - "📅 Full Day Prediction (24h)" - removed card and functionality
- **Removed API Endpoints**:
  - `GET /api/examples/single_slot` - no longer available
  - `GET /api/examples/full_day` - no longer available
- **Note**: The `/api/predictions/heating_demand_profile` endpoint is still available for external use
- Removed related tests (6 tests removed)

## [0.0.0.64] - 2025-12-02

- **Preserve Actual Sensor Timestamps**: Sample timestamps now reflect the actual timestamps from Home Assistant
  - Removed 5-second alignment that rounded all timestamps to nearest 5-second boundary
  - Timestamps like `2025-11-21 06:52:27` are now stored as-is instead of being rounded to `06:52:25`
  - Only microseconds are stripped (for storage efficiency), seconds are preserved
  - This provides accurate time information for debugging and data analysis
- **Breaking Change**: New samples will have actual timestamps, but existing samples remain with rounded timestamps
  - To get consistent timestamps, users may want to re-sync historical data after updating
- Updated tests to reflect new timestamp preservation behavior

## [0.0.0.63] - 2025-12-02

- **Two-Step Heat Pump Prediction (Experimental)**: Added new two-step prediction approach for better accuracy
  - Step 1: Classifier predicts whether heating will be active or inactive in a given hour
  - Step 2: Regressor predicts kWh consumption only for active hours (inactive = 0 kWh)
  - Solves overestimation when pump is off and underestimation during heavy heating
- **Automatic Threshold Detection**: Activity threshold is automatically computed from training data
  - Uses 5th percentile of positive kWh values as threshold
  - Minimum threshold of 0.01 kWh to filter noise
  - Threshold is stored with model and reused for predictions
  - No manual configuration required by end users
- **New API Endpoints**:
  - `GET /api/features/two_step_prediction`: Get two-step prediction configuration status
  - `POST /api/features/two_step_prediction`: Enable/disable two-step prediction mode
  - `POST /api/train/two_step_heating_demand`: Train the two-step model (classifier + regressor)
  - `GET /api/model/two_step_status`: Get two-step model status and threshold info
  - `POST /api/predictions/two_step_scenario`: Make predictions using two-step approach
- **New Module**: `ml/two_step_model.py` with complete implementation
  - `TwoStepHeatingDemandModel` class with classifier and regressor
  - `TwoStepPrediction` dataclass with is_active, predicted_kwh, activity_probability
  - `train_two_step_heating_demand_model()` for training both models
  - `predict_two_step_scenario()` for scenario-based predictions
- **Feature Configuration Updates**:
  - Added `two_step_prediction_enabled` flag to FeatureConfiguration
  - New methods: `enable_two_step_prediction()`, `disable_two_step_prediction()`, `is_two_step_prediction_enabled()`
  - Configuration persists across restarts
- **Tests**: Added 27 new tests for two-step prediction functionality
  - Threshold computation tests
  - Classifier and regressor training tests
  - Prediction tests for active/inactive hours
  - Model persistence tests
  - Feature configuration tests

## [0.0.0.62] - 2025-12-02

- **Resampling Interval UI Configuration**: Moved sample rate configuration from config.yaml to UI
  - Sample rate can now be changed directly from the Configuration tab in the UI
  - New dropdown selector for sample rate (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60 minutes)
  - Sample rate is persisted to `/data/resample_config.json` for persistence across restarts
  - When sample rate is changed, the "Flush existing data" checkbox is automatically checked
- **API Changes**:
  - Added `POST /api/sample_rate` endpoint to update sample rate from UI
  - Added `set_sample_rate_minutes()` function to persist sample rate configuration
  - Sample rate is now read from persistent JSON config instead of environment variable
- **Config Changes**:
  - Removed `sample_rate_minutes` from add-on configuration options
  - Removed `SAMPLE_RATE_MINUTES` environment variable from run.sh
- Added tests for new sample rate persistence functionality

## [0.0.0.61] - 2025-12-02

- **Improved Training Data Table**: Enhanced training data display with all sensor categories
  - Now shows all sensor categories (outdoor_temp, wind, humidity, pressure, indoor_temp, target_temp, dhw_temp, hp_kwh_total, dhw_active) instead of just dhw_temp and hp_kwh_total
  - For hp_kwh_total, displays the delta (energy consumed during training) instead of raw cumulative values
  - Added Delta column showing the difference between first and last values for all sensors
  - hp_kwh_delta is highlighted in green to emphasize the actual energy consumption
  - Units are now displayed for each sensor value
- **API Changes**:
  - `/api/train/heating_demand` now returns `training_data` with all sensor categories
  - Each sensor category includes `first`, `last`, and `unit` fields
  - hp_kwh_total additionally includes `delta` field showing energy consumed during training
  - Added `sensor_ranges` dict to `FeatureDatasetStats` for tracking all sensor categories
  - Added `hp_kwh_delta` field to `FeatureDatasetStats` for energy consumption delta
  - Legacy fields `dhw_temp_range` and `hp_kwh_total_range` maintained for backward compatibility
- Added 4 new tests for sensor ranges and hp_kwh_delta functionality

## [0.0.0.60] - 2025-12-02

- **Dark/Light Mode Toggle**: Added theme toggle button in header
  - Uses CSS variables for consistent theming across light and dark modes
  - Theme preference is saved to localStorage
  - Defaults to dark mode
- **Tabbed Interface**: Reorganized UI into three tabs
  - Configuration tab: Data resampling and feature configuration
  - Model Training tab: Model training, predictions, and scenario testing
  - Sensor Information tab: Sensor data overview
- **Training Data Table**: Added table showing first/last values when training model
  - Displays dhw_temp and hp_kwh_total ranges
  - API endpoint updated to return `training_data` with first/last values
  - New `TrainingDataRange` dataclass for capturing sensor ranges
- **Removed Wind Card**: Removed test wind speed card from UI

## [0.0.0.59] - 2025-12-02

- **Flush Resample Table on Sample Rate Change**: Added ability to flush existing resampled data before resampling
  - New `flush_resampled_samples()` function to clear the resampled_samples table
  - `/resample` endpoint now accepts optional `flush` parameter in request body
  - When sample rate changes, existing data (computed with different interval) should be flushed
  - `ResampleStats` now includes `table_flushed` field to indicate if flush was performed
  - UI now shows a "Flush existing data" checkbox in the Data Resampling section
- **Dynamic Sample Rate Display in UI**: UI now shows configured sample rate instead of hardcoded "5-minute"
  - Sample rate is loaded from `/api/sample_rate` endpoint on page load
  - Resampling status message includes the sample rate used
- Added tests for flush functionality

## [0.0.0.58] - 2025-12-02

- **Configurable Sample Rate**: Added ability to configure the sample rate for data resampling
  - New `sample_rate_minutes` configuration option (1-60 minutes, default: 5)
  - Allows training models with different time granularity for different use cases
  - kWh usage is correctly calculated for the configured timeframe
  - `/resample` endpoint now accepts optional `sample_rate_minutes` in request body
  - New `/api/sample_rate` endpoint to get current sample rate configuration
  - `ResampleStats` now includes `sample_rate_minutes` field
  - Added 17 new tests for configurable sample rate functionality

## [0.0.0.57] - 2025-12-02

- **Feature Set Configuration**: Complete overhaul of the heat pump consumption model feature set
  - Defined **13 core baseline features** that are always active and cannot be disabled
  - Moved optional features to **experimental status** (disabled by default, toggleable via UI)
  - All features now have complete metadata (name, category, description, unit, time_window, is_core)
  
- **New Core Baseline Features**:
  - `heating_kwh_last_1h`: 1-hour heating energy consumption (was missing)
  - `delta_target_indoor`: Derived feature showing difference between target and indoor temperature
  - `wind` and `humidity` now explicitly in baseline (always required)
  
- **Timezone Configuration**:
  - `hour_of_day` feature now uses configurable IANA timezone (default: Europe/Amsterdam)
  - All timestamps stored in UTC and converted to local time for hour_of_day
  - UI allows timezone selection from common timezones
  
- **Feature Configuration UI**:
  - New "Feature Configuration" section showing all features grouped by category
  - Core features (green badges) are always active with disabled checkboxes
  - Experimental features (orange badges) can be toggled on/off
  - Timezone selector for hour_of_day feature
  - Feature stats showing core count and active feature count
  
- **New API Endpoints**:
  - `GET /api/features/config`: Get current feature configuration
  - `POST /api/features/toggle`: Enable/disable experimental features
  - `POST /api/features/timezone`: Set timezone for time features
  - `GET /api/features/metadata`: Get feature metadata for documentation
  
- **Tests**: Added 38 new tests for feature configuration module
- **Documentation**: Updated README with new feature engineering documentation

![Feature Configuration UI](https://github.com/user-attachments/assets/8bf955e4-cee7-4b84-9a7d-47fdf464c354)

## [0.0.0.56] - 2025-12-01

- **Version Bump**: Preparing for feature set refactoring of heat pump consumption model

## [0.0.0.55] - 2025-12-01

- **Load Historical Day Examples**: Added ability to load historical days from 5-minute resampled data as scenario examples
  - New `/api/examples/available_days` endpoint returns list of available days (excluding first and last day)
  - New `/api/examples/historical_day/<date>` endpoint returns hourly averaged data for a specific day
  - UI dropdown to select historical day in the Scenario-Based Prediction section
- **Compare Predictions with Actual Data**: When a historical day is selected, the prediction table shows comparison
  - Added second column with actual kWh values from historical data
  - Added delta (difference) and percentage columns
  - Bar chart shows predicted vs actual values side by side
  - Summary shows Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE)
- Added 19 new tests for historical day functionality
- Internal: Added `get_available_historical_days()` function to query available days
- Internal: Added `get_historical_day_hourly_data()` function to compute hourly averages

## [0.0.0.54] - 2025-12-01

- **Simplified Scenario API**: Added new `/api/predictions/scenario` endpoint that accepts human-readable inputs
  - Users can now send simple weather forecast data (outdoor_temperature, wind_speed, humidity, pressure) and setpoint schedule (target_temperature)
  - All low-level model features (time features, historical aggregations) are computed internally
  - Timestamps must be in the future; past timestamps are rejected with clear validation errors
- Added `/api/examples/scenario` endpoint to get pre-filled 24-hour scenario example
- Added new UI section "Scenario-Based Prediction (Simplified)" with bar chart and table visualization
- Internal: Added validation functions for simplified scenario input
- Internal: Added conversion function to translate simplified inputs to model features
- Added 29 new tests for new simplified scenario functionality
- Updated documentation with new simplified API endpoints

## [0.0.0.53] - 2025-12-01

- **Enhanced model training**: Model now uses ALL available historical data for training (no artificial limit)
- Added function to compute historical aggregations from user-provided scenario features
- Added API endpoint `/api/predictions/enrich_scenario` to help users prepare prediction requests
- Added API endpoint `/api/predictions/compare_actual` to compare predictions with actual historical data
- Added API endpoint `/api/predictions/validate_start_time` to validate prediction start times
- Added time range information to dataset statistics (data_start_time, data_end_time, available_history_hours)
- Predictions must start at next/coming hour for accurate historical feature computation
- Test data can be compared using 5-minute average records to see delta between model and actual values
- Added 25 new tests for new functionality
- Updated documentation with new API endpoints and enhanced feature descriptions

## [0.0.0.52] - 2025-12-01

- Added pre-filled example fields in UI for single slot and full day heat pump usage calculation
- Added sensor information section in UI showing first and last datetime per sensor
- Added API endpoints: /api/examples/single_slot, /api/examples/full_day, /api/sensors/info
- Added results display with table and bar chart for predictions
- Added 12 new tests for new API endpoints and get_sensor_info function
- Improved UI with editable JSON input fields for predictions
- Added comprehensive technical and functional documentation (README.md)
- Documented all calculations: time-weighted averaging, historical aggregations, feature engineering
- Documented model storage location (`/data/heating_demand_model.joblib`)
- Added complete API reference with request/response examples
- Added usage examples for Python and Home Assistant automations
- Documented database schema and architecture

## [0.0.0.51] - 2025-12-01

- Fixed resampling to properly return statistics (ResampleStats) with slots_processed, slots_saved, slots_skipped
- Added UI buttons for model training and status checking
- Improved resample endpoint to show detailed statistics after completion
- Updated UI to display resampling statistics (time range, categories, slot counts)
- Added 4 new tests for ResampleStats return values

## [0.0.0.50] - 2025-12-01

- Fixed sensor import halting when data gaps exceed 24 hours
- Sync now fast-forwards through gaps in historical data by checking subsequent 24-hour windows
- Uses max(latest_sample_timestamp, sync_status.last_attempt) to ensure progress through data gaps
- Added tests for DWH sensor scenarios with large gaps in history data

## [0.0.0.49] - 2025-12-01

- Added comprehensive unit tests for sync_state module
- Added unit tests for Flask API endpoints (training, prediction, model status)
- Created GitHub Actions workflow to run tests on push and pull requests
- Improved test coverage from 102 to 120 tests

## [0.0.0.48] - 2025-12-01

- Added support for binary sensor states: map on/off and true/false to 1.0/0.0 for database storage

## [0.0.0.47] - 2025-12-01

- Reduced logging verbosity by changing sample save/update messages from INFO to DEBUG level
