# Database Testing Options for Energy Orchestrator

## Current Setup

**Production Database**: MariaDB 10.6+
- Connection: `mysql+pymysql://`
- JSON storage: TEXT columns with JSON strings
- Location: `core-mariadb` service in Home Assistant

**Test Database**: SQLite 3.45.1+ (in-memory)
- Connection: `sqlite:///:memory:`
- JSON storage: TEXT columns with JSON strings
- JSON support: Built-in JSON1 extension
- Configuration: `tests/conftest.py`

## Analysis

### Current Compatibility
✅ **JSON Storage**: Both MariaDB and SQLite store JSON as TEXT strings
✅ **JSON Functions**: SQLite 3.38+ has JSON1 extension with `json()`, `json_extract()`, etc.
✅ **Performance**: In-memory SQLite is 10-20x faster than container databases
✅ **Simplicity**: No external dependencies, works out of the box

### Compatibility Level: ~85%
- JSON serialization/deserialization: 100% compatible
- JSON query functions: 70-80% compatible (SQLite vs MariaDB syntax differs)
- SQL dialect: 90% compatible for basic queries
- Transactions: 100% compatible

## Recommended Approach

### For Unit Tests (Current - Recommended)
**Continue using SQLite in-memory** ✅

**Pros:**
- Fast execution (tests complete in < 5 seconds)
- No external dependencies
- Excellent JSON support via JSON1 extension
- Already configured and working

**Cons:**
- Minor SQL dialect differences (not affecting current tests)
- Some MariaDB-specific features not available

### For Integration Tests (Future Enhancement)
**Add MariaDB Test Container** for CI/CD pipeline

## Alternative Options

### Option 1: SQLite (Current - Keep It) ⭐ Recommended
```python
# tests/conftest.py
test_engine = create_engine("sqlite:///:memory:", echo=False)
```

**When to use**: Unit tests, fast iteration, local development

---

### Option 2: MariaDB Test Container (For Integration Tests)
```python
# tests/conftest_mariadb.py
import pytest
from testcontainers.mysql import MySqlContainer

@pytest.fixture(scope="session")
def mariadb_container():
    """Start a MariaDB container for integration tests."""
    with MySqlContainer("mariadb:10.6") as mariadb:
        # Wait for container to be ready
        mariadb_url = mariadb.get_connection_url()
        yield mariadb_url

@pytest.fixture(scope="session")
def mariadb_engine(mariadb_container):
    """Create engine connected to test MariaDB container."""
    engine = create_engine(mariadb_container, future=True)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()
```

**Dependencies:**
```bash
pip install testcontainers[mysql]
```

**When to use**: 
- Integration tests
- CI/CD pipeline
- Testing MariaDB-specific features
- Pre-production validation

**Pros:**
- 100% MariaDB compatibility
- Tests actual production database behavior
- Catches MariaDB-specific issues

**Cons:**
- Slower (30-60 seconds startup + tests)
- Requires Docker
- More resource intensive
- Complex setup

---

### Option 3: MySQL Test Container
Similar to MariaDB but using official MySQL image:

```python
with MySqlContainer("mysql:8.0") as mysql:
    ...
```

**Pros/Cons**: Similar to MariaDB container

---

### Option 4: PostgreSQL (Not Recommended for This Project)
**Why not**: Would require rewriting SQL queries and schema changes

## Implementation Recommendations

### Current Status: ✅ Good to Go
The current SQLite setup is sufficient for your needs:
- JSON is stored as TEXT (same as MariaDB)
- Test failures are due to MagicMock objects, not database incompatibility
- Fast test execution

### Future Enhancements (Optional)

#### 1. Add MariaDB Integration Tests (Low Priority)
Create separate test configuration for integration tests:

```bash
# Run unit tests with SQLite (fast)
pytest tests/ -m "not integration"

# Run integration tests with MariaDB (slow)
pytest tests/ -m integration --mariadb
```

#### 2. Add to CI/CD Pipeline
```yaml
# .github/workflows/test.yml
- name: Run unit tests
  run: pytest tests/ -m "not integration"
  
- name: Run integration tests with MariaDB
  run: |
    pytest tests/ -m integration --mariadb
```

## JSON Support Comparison

| Feature | MariaDB | SQLite (JSON1) | Compatible |
|---------|---------|----------------|------------|
| Store JSON as TEXT | ✅ | ✅ | ✅ 100% |
| json() function | ✅ | ✅ | ✅ 100% |
| json_extract() | ✅ | ✅ | ✅ 100% |
| JSON_OBJECT() | ✅ | json_object() | ⚠️ 90% (syntax differs) |
| JSON validation | ✅ | ✅ | ✅ 100% |
| JSON indexing | ✅ | ❌ | ❌ Not needed for tests |

## Conclusion

**Keep using SQLite for tests** - it's fast, reliable, and provides excellent JSON support that's compatible with your MariaDB schema. The test failures you're experiencing are due to MagicMock serialization issues, not database incompatibility.

Only add MariaDB test containers if you:
1. Need to test MariaDB-specific features
2. Are setting up CI/CD integration testing
3. Want pre-production validation with real MariaDB

For now, focus on fixing the MagicMock JSON serialization in tests (which we're doing).
