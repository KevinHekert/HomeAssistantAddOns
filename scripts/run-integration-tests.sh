#!/bin/bash
#
# Run integration tests with MariaDB in Docker
#
# Usage:
#   ./scripts/run-integration-tests.sh [options] [test_file]
#
# Options:
#   --keep-alive    Keep the test environment running after tests
#   --rebuild       Rebuild Docker images before running tests
#   --verbose       Show verbose test output
#
# Examples:
#   ./scripts/run-integration-tests.sh
#   ./scripts/run-integration-tests.sh --keep-alive
#   ./scripts/run-integration-tests.sh test_integration_samples.py
#   ./scripts/run-integration-tests.sh --rebuild --verbose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
KEEP_ALIVE=0
REBUILD=0
VERBOSE=""
TEST_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-alive)
            KEEP_ALIVE=1
            shift
            ;;
        --rebuild)
            REBUILD=1
            shift
            ;;
        --verbose)
            VERBOSE="-vv"
            shift
            ;;
        *)
            TEST_FILE="$1"
            shift
            ;;
    esac
done

echo -e "${BLUE}🧪 Energy Orchestrator Integration Tests${NC}"
echo -e "${BLUE}=========================================${NC}\n"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Navigate to repository root
cd "$(dirname "$0")/.."

# Rebuild if requested
if [ $REBUILD -eq 1 ]; then
    echo -e "${YELLOW}🔨 Rebuilding Docker images...${NC}"
    docker-compose -f docker-compose.test.yml build
    echo ""
fi

# Start test environment
echo -e "${YELLOW}🚀 Starting test environment...${NC}"
docker-compose -f docker-compose.test.yml up -d mariadb

# Wait for MariaDB to be healthy
echo -e "${YELLOW}⏳ Waiting for MariaDB to be ready...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker-compose -f docker-compose.test.yml exec -T mariadb mysqladmin ping -h localhost -u root -ptest_root_password &> /dev/null; then
        echo -e "${GREEN}✅ MariaDB is ready!${NC}\n"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -e "   Attempt $RETRY_COUNT/$MAX_RETRIES..."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ MariaDB failed to start after $MAX_RETRIES attempts${NC}"
    docker-compose -f docker-compose.test.yml logs mariadb
    docker-compose -f docker-compose.test.yml down -v
    exit 1
fi

# Display MariaDB version
echo -e "${BLUE}📊 MariaDB version:${NC}"
docker-compose -f docker-compose.test.yml exec -T mariadb mysql -u root -ptest_root_password -e "SELECT VERSION();" 2>/dev/null | grep -v "mysql: \[Warning\]"
echo ""

# Set test path
if [ -n "$TEST_FILE" ]; then
    TEST_PATH="tests/integration/$TEST_FILE"
    echo -e "${YELLOW}🎯 Running specific test: $TEST_FILE${NC}\n"
else
    TEST_PATH="tests/integration/"
    echo -e "${YELLOW}🎯 Running all integration tests${NC}\n"
fi

# Run integration tests
echo -e "${BLUE}🧪 Running integration tests...${NC}"
echo -e "${BLUE}================================${NC}\n"

export DB_HOST=localhost
export DB_PORT=3307
export DB_USER=energy_orchestrator
export DB_PASSWORD=test_password
export DB_NAME=energy_orchestrator_test
export PYTHONPATH="$(pwd)/energy_orchestrator/app"

cd energy_orchestrator/app

# Run pytest
if python -m pytest $TEST_PATH \
    -m integration \
    --tb=short \
    --timeout=300 \
    --cov=. \
    --cov-report=term-missing \
    --cov-report=html \
    $VERBOSE; then
    
    echo -e "\n${GREEN}✅ All integration tests passed!${NC}\n"
    TEST_EXIT_CODE=0
else
    echo -e "\n${RED}❌ Some integration tests failed${NC}\n"
    TEST_EXIT_CODE=1
fi

cd ../..

# Show coverage summary if available
if [ -f "energy_orchestrator/app/htmlcov/index.html" ]; then
    echo -e "${BLUE}📊 Coverage report generated: energy_orchestrator/app/htmlcov/index.html${NC}"
fi

# Cleanup or keep alive
if [ $KEEP_ALIVE -eq 1 ]; then
    echo -e "\n${YELLOW}⚠️  Test environment is still running (--keep-alive)${NC}"
    echo -e "${YELLOW}   To stop it manually, run:${NC}"
    echo -e "${YELLOW}   docker-compose -f docker-compose.test.yml down -v${NC}\n"
else
    echo -e "\n${YELLOW}🧹 Cleaning up test environment...${NC}"
    docker-compose -f docker-compose.test.yml down -v
    echo -e "${GREEN}✅ Cleanup complete${NC}\n"
fi

exit $TEST_EXIT_CODE
