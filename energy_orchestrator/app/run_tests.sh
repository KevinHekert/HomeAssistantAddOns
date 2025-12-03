#!/bin/bash
# Test runner script for Energy Orchestrator
# Usage: ./run_tests.sh [fast|full|file]

set -e

cd "$(dirname "$0")"

echo "Energy Orchestrator Test Runner"
echo "================================"

# Check if pytest is installed
if ! python -m pytest --version > /dev/null 2>&1; then
    echo "Installing pytest..."
    pip install pytest
fi

case "${1:-fast}" in
    fast)
        echo "Running fast tests (skipping optimizer tests)..."
        python -m pytest tests/ -v -k "not optimizer" --tb=short
        ;;
    full)
        echo "Running all tests (including slow optimizer tests)..."
        python -m pytest tests/ -v --tb=short
        ;;
    coverage)
        echo "Running tests with coverage..."
        pip install pytest-cov > /dev/null 2>&1
        python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term
        echo "Coverage report generated in htmlcov/index.html"
        ;;
    *)
        echo "Running tests from: $1"
        python -m pytest "$1" -v --tb=short
        ;;
esac

echo ""
echo "Test run complete!"
