"""
Integration tests for sample data CRUD operations.

These tests verify that sample data can be correctly inserted, queried,
updated, and deleted from the MariaDB database.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from db.models import Sample, SyncStatus


@pytest.mark.integration
def test_insert_sample(mariadb_engine, clean_database):
    """Test inserting a single sample into the database."""
    session = Session(mariadb_engine)
    
    now = datetime.utcnow()
    sample = Sample(
        entity_id="sensor.test_temperature",
        timestamp=now,
        value=21.5,
        unit="°C"
    )
    
    session.add(sample)
    session.commit()
    
    # Verify the sample was inserted
    retrieved = session.query(Sample).filter_by(entity_id="sensor.test_temperature").first()
    assert retrieved is not None
    assert retrieved.value == 21.5
    assert retrieved.unit == "°C"
    assert abs((retrieved.timestamp - now).total_seconds()) < 1
    
    session.close()


@pytest.mark.integration
def test_insert_multiple_samples(mariadb_engine, clean_database):
    """Test bulk insert of multiple samples."""
    session = Session(mariadb_engine)
    
    now = datetime.utcnow()
    samples = []
    
    for i in range(100):
        samples.append(Sample(
            entity_id="sensor.test_temperature",
            timestamp=now + timedelta(minutes=i),
            value=20.0 + i * 0.1,
            unit="°C"
        ))
    
    session.add_all(samples)
    session.commit()
    
    # Verify all samples were inserted
    count = session.query(Sample).filter_by(entity_id="sensor.test_temperature").count()
    assert count == 100
    
    # Verify ordering
    retrieved = session.query(Sample).filter_by(
        entity_id="sensor.test_temperature"
    ).order_by(Sample.timestamp).all()
    
    assert len(retrieved) == 100
    assert retrieved[0].value == 20.0
    assert retrieved[99].value == pytest.approx(29.9, rel=1e-5)
    
    session.close()


@pytest.mark.integration
def test_query_samples_by_timerange(mariadb_engine, sample_test_data):
    """Test querying samples within a specific time range."""
    session = sample_test_data
    
    now = datetime.utcnow()
    start_time = now - timedelta(hours=6)
    end_time = now - timedelta(hours=3)
    
    # Query samples in time range
    samples = session.query(Sample).filter(
        Sample.entity_id == "sensor.smile_outdoor_temperature",
        Sample.timestamp >= start_time,
        Sample.timestamp <= end_time
    ).order_by(Sample.timestamp).all()
    
    print(f"\n📊 Found {len(samples)} samples in 3-hour window")
    
    # Should have samples in this range
    assert len(samples) > 0
    
    # All samples should be within range
    for sample in samples:
        assert start_time <= sample.timestamp <= end_time


@pytest.mark.integration
def test_query_latest_sample(mariadb_engine, sample_test_data):
    """Test retrieving the most recent sample for an entity."""
    session = sample_test_data
    
    latest = session.query(Sample).filter_by(
        entity_id="sensor.anna_temperature"
    ).order_by(Sample.timestamp.desc()).first()
    
    assert latest is not None
    print(f"\n🌡️ Latest indoor temp: {latest.value}°C at {latest.timestamp}")


@pytest.mark.integration
def test_query_with_aggregation(mariadb_engine, sample_test_data):
    """Test aggregation queries on sample data."""
    from sqlalchemy import func
    
    session = sample_test_data
    
    # Calculate average temperature
    avg_temp = session.query(
        func.avg(Sample.value)
    ).filter_by(
        entity_id="sensor.smile_outdoor_temperature"
    ).scalar()
    
    # Calculate min and max
    min_temp = session.query(
        func.min(Sample.value)
    ).filter_by(
        entity_id="sensor.smile_outdoor_temperature"
    ).scalar()
    
    max_temp = session.query(
        func.max(Sample.value)
    ).filter_by(
        entity_id="sensor.smile_outdoor_temperature"
    ).scalar()
    
    print(f"\n📈 Temperature stats: min={min_temp:.1f}°C, avg={avg_temp:.1f}°C, max={max_temp:.1f}°C")
    
    assert min_temp < avg_temp < max_temp
    assert 0 < avg_temp < 50  # Reasonable outdoor temperature range


@pytest.mark.integration
def test_update_sample(mariadb_engine, clean_database):
    """Test updating an existing sample."""
    session = Session(mariadb_engine)
    
    now = datetime.utcnow()
    sample = Sample(
        entity_id="sensor.test_temperature",
        timestamp=now,
        value=21.5,
        unit="°C"
    )
    
    session.add(sample)
    session.commit()
    
    # Update the value
    sample.value = 22.0
    session.commit()
    
    # Verify the update
    retrieved = session.query(Sample).filter_by(
        entity_id="sensor.test_temperature",
        timestamp=now
    ).first()
    
    assert retrieved.value == 22.0
    
    session.close()


@pytest.mark.integration
def test_delete_sample(mariadb_engine, clean_database):
    """Test deleting a sample."""
    session = Session(mariadb_engine)
    
    now = datetime.utcnow()
    sample = Sample(
        entity_id="sensor.test_temperature",
        timestamp=now,
        value=21.5,
        unit="°C"
    )
    
    session.add(sample)
    session.commit()
    
    # Verify it exists
    count_before = session.query(Sample).filter_by(entity_id="sensor.test_temperature").count()
    assert count_before == 1
    
    # Delete it
    session.delete(sample)
    session.commit()
    
    # Verify it's gone
    count_after = session.query(Sample).filter_by(entity_id="sensor.test_temperature").count()
    assert count_after == 0
    
    session.close()


@pytest.mark.integration
def test_delete_old_samples(mariadb_engine, sample_test_data):
    """Test bulk deletion of old samples."""
    session = sample_test_data
    
    # Count samples before deletion
    count_before = session.query(Sample).filter_by(
        entity_id="sensor.smile_outdoor_temperature"
    ).count()
    
    # Delete samples older than 12 hours
    cutoff = datetime.utcnow() - timedelta(hours=12)
    deleted = session.query(Sample).filter(
        Sample.entity_id == "sensor.smile_outdoor_temperature",
        Sample.timestamp < cutoff
    ).delete()
    
    session.commit()
    
    print(f"\n🗑️ Deleted {deleted} samples older than 12 hours")
    
    # Verify deletion
    count_after = session.query(Sample).filter_by(
        entity_id="sensor.smile_outdoor_temperature"
    ).count()
    
    assert count_after == count_before - deleted
    assert deleted > 0


@pytest.mark.integration
def test_sync_status_tracking(mariadb_engine, clean_database):
    """Test sync status tracking for entities."""
    session = Session(mariadb_engine)
    
    entity_id = "sensor.test_sensor"
    now = datetime.utcnow()
    
    # Create sync status
    status = SyncStatus(
        entity_id=entity_id,
        last_attempt=now,
        last_success=now
    )
    
    session.add(status)
    session.commit()
    
    # Retrieve and verify
    retrieved = session.query(SyncStatus).filter_by(entity_id=entity_id).first()
    assert retrieved is not None
    assert abs((retrieved.last_success - now).total_seconds()) < 1
    
    # Update sync status
    new_time = now + timedelta(minutes=5)
    retrieved.last_attempt = new_time
    session.commit()
    
    # Verify update
    updated = session.query(SyncStatus).filter_by(entity_id=entity_id).first()
    assert abs((updated.last_attempt - new_time).total_seconds()) < 1
    
    session.close()


@pytest.mark.integration
def test_concurrent_inserts(mariadb_engine, clean_database):
    """Test that concurrent inserts work correctly (using separate sessions)."""
    now = datetime.utcnow()
    
    # Session 1: Insert sample for sensor A
    session1 = Session(mariadb_engine)
    sample1 = Sample(
        entity_id="sensor.a",
        timestamp=now,
        value=10.0,
        unit="test"
    )
    session1.add(sample1)
    session1.commit()
    session1.close()
    
    # Session 2: Insert sample for sensor B
    session2 = Session(mariadb_engine)
    sample2 = Sample(
        entity_id="sensor.b",
        timestamp=now,
        value=20.0,
        unit="test"
    )
    session2.add(sample2)
    session2.commit()
    session2.close()
    
    # Verify both samples exist
    session3 = Session(mariadb_engine)
    count = session3.query(Sample).filter(
        Sample.entity_id.in_(["sensor.a", "sensor.b"])
    ).count()
    assert count == 2
    session3.close()
