#!/bin/bash
#
# Setup and initialize the test database
#
# This script:
# 1. Starts the MariaDB test container
# 2. Waits for it to be ready
# 3. Runs initialization scripts
# 4. Verifies the setup
#
# Usage:
#   ./scripts/setup-test-db.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🗄️  Setting up Test Database${NC}"
echo -e "${BLUE}============================${NC}\n"

# Navigate to repository root
cd "$(dirname "$0")/.."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Start MariaDB container
echo -e "${YELLOW}🚀 Starting MariaDB container...${NC}"
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
    exit 1
fi

# Display database information
echo -e "${BLUE}📊 Database Information:${NC}"
docker-compose -f docker-compose.test.yml exec -T mariadb mysql -u root -ptest_root_password -e "SELECT VERSION() AS 'Version', DATABASE() AS 'Current Database';" 2>/dev/null | grep -v "mysql: \[Warning\]"
echo ""

# Verify database exists
echo -e "${BLUE}📋 Verifying database setup:${NC}"
docker-compose -f docker-compose.test.yml exec -T mariadb mysql -u root -ptest_root_password -e "SHOW DATABASES;" 2>/dev/null | grep -v "mysql: \[Warning\]"
echo ""

# Test connection with application user
echo -e "${BLUE}🔐 Testing application user connection:${NC}"
if docker-compose -f docker-compose.test.yml exec -T mariadb mysql -h localhost -u energy_orchestrator -ptest_password energy_orchestrator_test -e "SELECT 'Connection successful' AS status;" 2>/dev/null | grep -v "mysql: \[Warning\]"; then
    echo -e "${GREEN}✅ Application user can connect${NC}\n"
else
    echo -e "${RED}❌ Application user connection failed${NC}\n"
    exit 1
fi

# Display connection info
echo -e "${BLUE}📝 Connection Details:${NC}"
echo -e "   Host: localhost"
echo -e "   Port: 3307"
echo -e "   Database: energy_orchestrator_test"
echo -e "   User: energy_orchestrator"
echo -e "   Password: test_password"
echo ""

echo -e "${GREEN}✅ Test database setup complete!${NC}"
echo -e "${YELLOW}   The database is now running and ready for integration tests.${NC}"
echo -e "${YELLOW}   To stop it, run: docker-compose -f docker-compose.test.yml down -v${NC}\n"
