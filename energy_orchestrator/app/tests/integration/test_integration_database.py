"""
Integration tests for database connectivity and basic operations.

These tests verify that the MariaDB connection works correctly and
basic database operations function as expected.
"""

import pytest
from sqlalchemy import text
from datetime import datetime


@pytest.mark.integration
def test_mariadb_connection(mariadb_engine):
    """Test that we can connect to MariaDB."""
    with mariadb_engine.connect() as conn:
        result = conn.execute(text("SELECT 1 AS test"))
        assert result.scalar() == 1


@pytest.mark.integration
def test_database_version(mariadb_engine):
    """Test that we're using the correct MariaDB version."""
    with mariadb_engine.connect() as conn:
        result = conn.execute(text("SELECT VERSION()"))
        version = result.scalar()
        
        print(f"\n📊 MariaDB version: {version}")
        assert "MariaDB" in version or "MySQL" in version
        
        # Check that we're using at least MariaDB 10.6
        if "MariaDB" in version:
            version_parts = version.split("-")[0].split(".")
            major = int(version_parts[0])
            minor = int(version_parts[1])
            assert major >= 10 and minor >= 6, f"MariaDB version {major}.{minor} is too old (need 10.6+)"


@pytest.mark.integration
def test_database_tables_created(mariadb_engine):
    """Test that all required tables exist."""
    expected_tables = [
        "samples",
        "sync_status",
        "sensor_mappings",
        "resampled_samples",
        "feature_statistics",
    ]
    
    with mariadb_engine.connect() as conn:
        result = conn.execute(text("SHOW TABLES"))
        tables = [row[0] for row in result]
        
        print(f"\n📋 Found tables: {tables}")
        
        for expected in expected_tables:
            assert expected in tables, f"Table '{expected}' not found in database"


@pytest.mark.integration
def test_database_charset(mariadb_engine):
    """Test that database uses UTF-8 charset."""
    with mariadb_engine.connect() as conn:
        result = conn.execute(text(
            "SELECT DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME "
            "FROM INFORMATION_SCHEMA.SCHEMATA "
            "WHERE SCHEMA_NAME = DATABASE()"
        ))
        charset, collation = result.fetchone()
        
        print(f"\n🔤 Charset: {charset}, Collation: {collation}")
        assert charset in ["utf8", "utf8mb4"], f"Expected UTF-8, got {charset}"


@pytest.mark.integration
def test_json_support(mariadb_engine):
    """Test that MariaDB supports JSON operations."""
    with mariadb_engine.connect() as conn:
        # Test JSON_OBJECT function
        result = conn.execute(text(
            "SELECT JSON_OBJECT('key', 'value') AS json_data"
        ))
        json_data = result.scalar()
        
        print(f"\n📦 JSON test result: {json_data}")
        assert "key" in json_data and "value" in json_data


@pytest.mark.integration
def test_datetime_handling(mariadb_engine):
    """Test that datetime handling works correctly."""
    with mariadb_engine.connect() as conn:
        now = datetime.utcnow()
        
        # Insert a test timestamp
        conn.execute(text(
            "CREATE TEMPORARY TABLE test_datetime (id INT PRIMARY KEY, dt DATETIME)"
        ))
        conn.execute(
            text("INSERT INTO test_datetime (id, dt) VALUES (:id, :dt)"),
            {"id": 1, "dt": now}
        )
        
        # Retrieve it back
        result = conn.execute(text("SELECT dt FROM test_datetime WHERE id = 1"))
        retrieved = result.scalar()
        
        # Check that the datetime is correct (within 1 second due to potential precision loss)
        time_diff = abs((retrieved - now).total_seconds())
        assert time_diff < 1, f"Datetime mismatch: {now} vs {retrieved}"


@pytest.mark.integration
def test_transaction_rollback(mariadb_engine):
    """Test that transaction rollback works correctly."""
    from db.models import Sample
    
    with mariadb_engine.connect() as conn:
        # Start a transaction
        trans = conn.begin()
        
        try:
            # Insert a sample
            conn.execute(text(
                "INSERT INTO samples (entity_id, timestamp, value, unit) "
                "VALUES ('test.sensor', NOW(), 42.0, 'test')"
            ))
            
            # Check it's there
            result = conn.execute(text(
                "SELECT COUNT(*) FROM samples WHERE entity_id = 'test.sensor'"
            ))
            assert result.scalar() == 1
            
            # Rollback
            trans.rollback()
        except:
            trans.rollback()
            raise
    
    # Verify the sample was rolled back
    with mariadb_engine.connect() as conn:
        result = conn.execute(text(
            "SELECT COUNT(*) FROM samples WHERE entity_id = 'test.sensor'"
        ))
        assert result.scalar() == 0


@pytest.mark.integration
def test_unique_constraint_enforcement(mariadb_engine, clean_database):
    """Test that unique constraints are enforced."""
    from sqlalchemy.exc import IntegrityError
    from db.models import Sample
    from sqlalchemy.orm import Session
    
    session = Session(mariadb_engine)
    
    # Create a sample
    now = datetime.utcnow()
    sample1 = Sample(
        entity_id="test.sensor",
        timestamp=now,
        value=42.0,
        unit="test"
    )
    session.add(sample1)
    session.commit()
    
    # Try to create a duplicate (same entity_id and timestamp)
    sample2 = Sample(
        entity_id="test.sensor",
        timestamp=now,
        value=99.0,
        unit="test"
    )
    session.add(sample2)
    
    with pytest.raises(IntegrityError):
        session.commit()
    
    session.rollback()
    session.close()


@pytest.mark.integration
def test_database_performance_simple_query(mariadb_engine, clean_database):
    """Test basic query performance."""
    import time
    from db.models import Sample
    from sqlalchemy.orm import Session
    
    session = Session(mariadb_engine)
    
    # Insert 1000 samples
    now = datetime.utcnow()
    samples = []
    for i in range(1000):
        samples.append(Sample(
            entity_id="test.sensor",
            timestamp=now.replace(microsecond=i),
            value=float(i),
            unit="test"
        ))
    
    session.add_all(samples)
    session.commit()
    
    # Time a simple query
    start = time.time()
    result = session.query(Sample).filter(Sample.entity_id == "test.sensor").count()
    duration = time.time() - start
    
    print(f"\n⏱️ Query for 1000 records took {duration:.3f} seconds")
    
    assert result == 1000
    assert duration < 1.0, f"Query too slow: {duration:.3f}s"
    
    session.close()
