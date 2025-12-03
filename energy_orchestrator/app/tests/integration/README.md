# Integration Tests for Energy Orchestrator

This directory contains integration tests that run against a real MariaDB database.

## Overview

Integration tests validate the entire system working together:
- Database interactions with MariaDB
- Data persistence and retrieval
- Complex workflows across multiple modules
- Real SQL queries and transactions

## Differences from Unit Tests

| Aspect | Unit Tests | Integration Tests |
|--------|-----------|-------------------|
| Database | SQLite in-memory | MariaDB container |
| Speed | Fast (< 5s) | Slower (~30-60s) |
| Scope | Individual functions | Full workflows |
| Dependencies | Mocked | Real services |

## Running Integration Tests

### Prerequisites

1. Docker and Docker Compose installed
2. Python 3.12+ installed

### Local Development

```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Wait for services to be healthy
docker-compose -f docker-compose.test.yml ps

# Run integration tests
cd energy_orchestrator/app
pytest tests/integration/ -v -m integration

# Stop test environment
docker-compose -f docker-compose.test.yml down -v
```

### Using the convenience script

```bash
# Run all integration tests
./scripts/run-integration-tests.sh

# Run specific test file
./scripts/run-integration-tests.sh test_integration_samples.py

# Keep environment running after tests
./scripts/run-integration-tests.sh --keep-alive
```

## Test Structure

- `conftest_integration.py` - Pytest fixtures for MariaDB integration
- `test_integration_database.py` - Database connectivity tests
- `test_integration_samples.py` - Sample data CRUD operations
- `test_integration_resampling.py` - Data resampling workflow
- `test_integration_feature_stats.py` - Feature statistics calculation
- `test_integration_full_workflow.py` - End-to-end workflows

## Writing Integration Tests

Mark tests with the `integration` marker:

```python
import pytest

@pytest.mark.integration
def test_something(integration_session):
    """Test with real MariaDB database"""
    # Your test code here
    pass
```

### Available Fixtures

- `mariadb_engine` - SQLAlchemy engine connected to MariaDB
- `integration_session` - Database session with automatic rollback
- `clean_database` - Clean all tables before test
- `sample_test_data` - Pre-populated test data

## Environment Variables

Configure the test database connection:

```bash
export DB_HOST=localhost
export DB_PORT=3307
export DB_USER=energy_orchestrator
export DB_PASSWORD=test_password
export DB_NAME=energy_orchestrator_test
```

## GitHub Actions

Integration tests run automatically in CI via `.github/workflows/integration-tests.yml`.

## Troubleshooting

### Database connection fails
```bash
# Check if MariaDB is running
docker-compose -f docker-compose.test.yml ps mariadb

# View logs
docker-compose -f docker-compose.test.yml logs mariadb
```

### Port conflicts
If port 3307 is in use, change it in `docker-compose.test.yml`:
```yaml
ports:
  - "3308:3306"  # Use different port
```

Then update `DB_PORT` environment variable.

### Permission issues
```bash
# Reset database volume
docker-compose -f docker-compose.test.yml down -v
docker-compose -f docker-compose.test.yml up -d
```
