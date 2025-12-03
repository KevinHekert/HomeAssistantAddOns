# Integration Testing Guide

This document provides comprehensive guidance for running and maintaining integration tests for the Energy Orchestrator add-on with MariaDB.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [GitHub Actions](#github-actions)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

### Purpose

Integration tests verify that the Energy Orchestrator add-on works correctly with a real MariaDB database, as it would in production. Unlike unit tests that use SQLite in-memory databases, integration tests:

- Use **MariaDB 10.6+** (matching production)
- Test **real SQL queries and transactions**
- Verify **complete workflows** end-to-end
- Catch **database-specific issues** early

### When to Run Integration Tests

- Before major releases
- When modifying database schemas or queries
- When adding new database-dependent features
- In CI/CD pipeline for pull requests
- When troubleshooting production database issues

---

## Architecture

### Test Environment Components

The integration test environment consists of:

1. **MariaDB Container** (mariadb:10.6)
   - Production-parity database
   - Pre-configured with test credentials
   - Isolated test database

2. **Mock Home Assistant API** (Flask)
   - Simulates HA API endpoints
   - Provides mock sensor data
   - Enables end-to-end testing

3. **Test Framework** (pytest)
   - pytest fixtures for database setup
   - Automatic cleanup between tests
   - Coverage reporting

### Directory Structure

```
HomeAssistantAddOns/
├── docker-compose.test.yml           # Test environment definition
├── scripts/
│   ├── run-integration-tests.sh      # Main test runner script
│   ├── setup-test-db.sh              # Database setup script
│   └── init-test-db.sql              # DB initialization SQL
├── tests/
│   └── mock_homeassistant/           # Mock HA API server
│       ├── app.py
│       ├── Dockerfile
│       └── README.md
└── energy_orchestrator/
    └── app/
        └── tests/
            └── integration/          # Integration test suite
                ├── conftest_integration.py
                ├── test_integration_database.py
                ├── test_integration_samples.py
                ├── test_integration_resampling.py
                ├── test_integration_feature_stats.py
                └── test_integration_full_workflow.py
```

---

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.12+
- Bash shell (Linux/macOS/WSL)

### Run All Integration Tests

```bash
./scripts/run-integration-tests.sh
```

This script will:
1. Start MariaDB container
2. Wait for database to be ready
3. Run all integration tests
4. Generate coverage report
5. Clean up environment

---

## Detailed Setup

### 1. Environment Setup

Install required Python packages:

```bash
pip install -r energy_orchestrator/requirements.txt
pip install pytest pytest-cov pytest-timeout
```

### 2. Start Test Database

```bash
./scripts/setup-test-db.sh
```

Or manually with Docker Compose:

```bash
docker-compose -f docker-compose.test.yml up -d mariadb
```

### 3. Verify Database Connection

```bash
mysql -h 127.0.0.1 -P 3307 -u energy_orchestrator -ptest_password energy_orchestrator_test -e "SELECT 'Success';"
```

---

## Running Tests

### Run All Integration Tests

```bash
./scripts/run-integration-tests.sh
```

### Run Specific Test File

```bash
./scripts/run-integration-tests.sh test_integration_samples.py
```

### Run with Options

```bash
# Keep environment running after tests (for debugging)
./scripts/run-integration-tests.sh --keep-alive

# Rebuild Docker images before running
./scripts/run-integration-tests.sh --rebuild

# Verbose output
./scripts/run-integration-tests.sh --verbose
```

### Manual Test Execution

```bash
# Set environment variables
export DB_HOST=localhost
export DB_PORT=3307
export DB_USER=energy_orchestrator
export DB_PASSWORD=test_password
export DB_NAME=energy_orchestrator_test
export PYTHONPATH="$(pwd)/energy_orchestrator/app"

# Run tests
cd energy_orchestrator/app
python -m pytest tests/integration/ -v -m integration
```

---

## Writing Tests

### Test Structure

All integration tests should:

1. Be marked with `@pytest.mark.integration`
2. Use the provided fixtures from `conftest_integration.py`
3. Include descriptive docstrings
4. Print progress messages for visibility

### Example Test

```python
import pytest
from datetime import datetime
from sqlalchemy.orm import Session
from db.models import Sample

@pytest.mark.integration
def test_insert_sample(mariadb_engine, clean_database):
    """Test inserting a sample into MariaDB."""
    session = Session(mariadb_engine)
    
    sample = Sample(
        entity_id="sensor.test",
        timestamp=datetime.utcnow(),
        value=21.5,
        unit="°C"
    )
    
    session.add(sample)
    session.commit()
    
    # Verify
    retrieved = session.query(Sample).filter_by(
        entity_id="sensor.test"
    ).first()
    
    assert retrieved is not None
    assert retrieved.value == 21.5
    
    print(f"✅ Sample inserted: {retrieved.value}°C")
    
    session.close()
```

### Available Fixtures

#### `mariadb_engine`
- Scope: session
- Provides SQLAlchemy engine connected to MariaDB
- Automatically creates and drops schema

#### `integration_session`
- Scope: function
- Provides database session with automatic rollback
- Use for tests that need database access

#### `clean_database`
- Scope: function
- Truncates all tables before test
- Use for tests requiring isolated data

#### `sample_test_data`
- Scope: function
- Pre-populates database with sample data
- Includes sensors, mappings, and 24h of samples

---

## GitHub Actions

### Workflow Configuration

Integration tests run automatically in GitHub Actions on:
- Push to `main` or `master` branches
- Pull requests modifying Energy Orchestrator code
- Manual workflow dispatch

See `.github/workflows/integration-tests.yml` for configuration.

### CI Environment

GitHub Actions uses:
- MariaDB service container
- Same credentials as local testing
- Coverage reporting with artifacts

### Viewing Results

1. Go to **Actions** tab in GitHub
2. Select the workflow run
3. View test output in **Run integration tests** step
4. Download coverage reports from **Artifacts**

---

## Troubleshooting

### Database Connection Fails

**Problem:** Tests can't connect to MariaDB

**Solutions:**

1. Check if MariaDB is running:
   ```bash
   docker-compose -f docker-compose.test.yml ps
   ```

2. View MariaDB logs:
   ```bash
   docker-compose -f docker-compose.test.yml logs mariadb
   ```

3. Verify port availability:
   ```bash
   lsof -i :3307
   ```

4. Reset environment:
   ```bash
   docker-compose -f docker-compose.test.yml down -v
   ./scripts/setup-test-db.sh
   ```

### Tests Timeout

**Problem:** Tests hang or timeout

**Solutions:**

1. Increase timeout in pytest:
   ```bash
   pytest tests/integration/ --timeout=600
   ```

2. Check for deadlocks:
   ```bash
   docker-compose -f docker-compose.test.yml exec mariadb \
     mysql -u root -ptest_root_password -e "SHOW PROCESSLIST;"
   ```

### Permission Denied Errors

**Problem:** Can't execute scripts

**Solution:**
```bash
chmod +x scripts/*.sh
```

### Port Conflicts

**Problem:** Port 3307 already in use

**Solution:** Change port in `docker-compose.test.yml`:
```yaml
ports:
  - "3308:3306"  # Use different port
```

Then update `DB_PORT` environment variable.

### Container Won't Start

**Problem:** MariaDB container fails to start

**Solutions:**

1. Remove existing volumes:
   ```bash
   docker-compose -f docker-compose.test.yml down -v
   ```

2. Check disk space:
   ```bash
   df -h
   ```

3. View detailed logs:
   ```bash
   docker-compose -f docker-compose.test.yml up mariadb
   ```

---

## Best Practices

### Test Organization

1. **One concern per test** - Each test should verify one specific behavior
2. **Descriptive names** - Use clear, descriptive test function names
3. **Independent tests** - Tests should not depend on each other
4. **Clean state** - Use fixtures to ensure clean state

### Performance

1. **Use transactions** - Wrap operations in transactions when possible
2. **Limit data size** - Use minimal data needed for each test
3. **Skip expensive tests** - Mark slow tests for selective execution
4. **Parallel execution** - Consider pytest-xdist for parallel tests

### Data Management

1. **Use fixtures** - Leverage pytest fixtures for test data
2. **Clean up** - Fixtures handle cleanup automatically
3. **Realistic data** - Use realistic values similar to production
4. **Edge cases** - Test boundary conditions and error cases

### Debugging

1. **Print statements** - Use print() for debugging (shown in output)
2. **Keep environment** - Use `--keep-alive` to inspect database state
3. **SQL logging** - Enable SQLAlchemy echo for query debugging
4. **Breakpoints** - Use `pytest --pdb` for interactive debugging

---

## Environment Variables

### Required Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | localhost | MariaDB hostname |
| `DB_PORT` | 3307 | MariaDB port |
| `DB_USER` | energy_orchestrator | Database user |
| `DB_PASSWORD` | test_password | Database password |
| `DB_NAME` | energy_orchestrator_test | Database name |
| `PYTHONPATH` | (project root)/energy_orchestrator/app | Python module path |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTEST_TIMEOUT` | 300 | Test timeout in seconds |
| `SQLALCHEMY_ECHO` | False | Enable SQL query logging |

---

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [SQLAlchemy documentation](https://docs.sqlalchemy.org/)
- [MariaDB documentation](https://mariadb.org/documentation/)
- [Docker Compose documentation](https://docs.docker.com/compose/)

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. View logs: `docker-compose -f docker-compose.test.yml logs`
3. Open an issue on GitHub with detailed error information
