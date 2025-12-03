"""
End-to-end integration tests for complete workflows.

These tests verify that the entire system works together from
data ingestion through feature calculation and model training.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from db.models import (
    Sample, SensorMapping, ResampledSample,
    FeatureStatistic, SyncStatus
)


@pytest.mark.integration
def test_full_data_pipeline(mariadb_engine, clean_database):
    """
    Test the complete data pipeline:
    1. Insert raw samples
    2. Create sensor mappings
    3. Resample data
    4. Calculate feature statistics
    """
    session = Session(mariadb_engine)
    
    print("\n🔄 Starting full pipeline test...")
    
    # Step 1: Create sensor mappings
    print("  1️⃣ Creating sensor mappings...")
    mappings = [
        SensorMapping(
            category="outdoor_temp",
            entity_id="sensor.outdoor",
            is_active=True,
            priority=1
        ),
        SensorMapping(
            category="indoor_temp",
            entity_id="sensor.indoor",
            is_active=True,
            priority=1
        ),
    ]
    session.add_all(mappings)
    session.commit()
    
    # Step 2: Insert raw samples (24 hours of data)
    print("  2️⃣ Inserting raw samples...")
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    samples = []
    
    for i in range(288):  # 24 hours * 12 samples/hour
        timestamp = base_time + timedelta(minutes=i*5)
        samples.extend([
            Sample(
                entity_id="sensor.outdoor",
                timestamp=timestamp,
                value=10.0 + (i % 24),
                unit="°C"
            ),
            Sample(
                entity_id="sensor.indoor",
                timestamp=timestamp,
                value=21.0 + (i % 10) * 0.1,
                unit="°C"
            ),
        ])
    
    session.add_all(samples)
    session.commit()
    print(f"    ✅ Inserted {len(samples)} raw samples")
    
    # Step 3: Resample data
    print("  3️⃣ Resampling data...")
    from db.resample import resample_category
    
    # Resample first 10 slots
    for i in range(10):
        slot_start = base_time + timedelta(minutes=i*5)
        slot_end = slot_start + timedelta(minutes=5)
        resample_category("outdoor_temp", slot_start, slot_end)
        resample_category("indoor_temp", slot_start, slot_end)
    
    resampled_count = session.query(ResampledSample).count()
    print(f"    ✅ Created {resampled_count} resampled entries")
    
    # Step 4: Calculate feature statistics
    print("  4️⃣ Calculating feature statistics...")
    from db.calculate_feature_stats import calculate_feature_stats_for_slot
    
    target_slot = base_time + timedelta(minutes=9*5)
    calculate_feature_stats_for_slot(target_slot)
    
    stats_count = session.query(FeatureStatistic).filter_by(slot_start=target_slot).count()
    print(f"    ✅ Calculated {stats_count} feature statistics")
    
    # Verify the pipeline worked
    assert resampled_count >= 20  # At least 10 slots * 2 categories
    assert stats_count > 0
    
    print("\n✅ Full pipeline test completed successfully!")
    
    session.close()


@pytest.mark.integration
def test_continuous_data_flow(mariadb_engine, clean_database):
    """
    Test continuous data flow over multiple time periods.
    Simulates real-world scenario of data arriving periodically.
    """
    session = Session(mariadb_engine)
    
    print("\n🔄 Testing continuous data flow...")
    
    # Setup
    mapping = SensorMapping(
        category="outdoor_temp",
        entity_id="sensor.outdoor",
        is_active=True,
        priority=1
    )
    session.add(mapping)
    session.commit()
    
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # Simulate data arriving in batches
    for batch in range(5):
        print(f"  📦 Processing batch {batch + 1}/5...")
        
        # Add samples for this batch (1 hour each)
        samples = []
        batch_start = base_time + timedelta(hours=batch)
        
        for i in range(12):  # 12 samples per hour
            timestamp = batch_start + timedelta(minutes=i*5)
            samples.append(Sample(
                entity_id="sensor.outdoor",
                timestamp=timestamp,
                value=10.0 + batch + i * 0.1,
                unit="°C"
            ))
        
        session.add_all(samples)
        session.commit()
        
        # Resample this batch
        from db.resample import resample_category
        for i in range(12):
            slot_start = batch_start + timedelta(minutes=i*5)
            slot_end = slot_start + timedelta(minutes=5)
            resample_category("outdoor_temp", slot_start, slot_end)
        
        # Calculate stats for last slot in batch
        from db.calculate_feature_stats import calculate_feature_stats_for_slot
        last_slot = batch_start + timedelta(minutes=11*5)
        calculate_feature_stats_for_slot(last_slot)
    
    # Verify continuous processing
    total_samples = session.query(Sample).count()
    total_resampled = session.query(ResampledSample).count()
    total_stats = session.query(FeatureStatistic).count()
    
    print(f"\n📊 Pipeline results:")
    print(f"    Raw samples: {total_samples}")
    print(f"    Resampled: {total_resampled}")
    print(f"    Statistics: {total_stats}")
    
    assert total_samples == 60  # 5 batches * 12 samples
    assert total_resampled >= 60
    assert total_stats > 0
    
    print("✅ Continuous flow test passed!")
    
    session.close()


@pytest.mark.integration
def test_sync_status_workflow(mariadb_engine, clean_database):
    """
    Test the sync status tracking workflow.
    Verifies that sync status is properly maintained during data operations.
    """
    session = Session(mariadb_engine)
    
    print("\n🔄 Testing sync status workflow...")
    
    entity_id = "sensor.test_sensor"
    now = datetime.utcnow()
    
    # Initial sync attempt
    print("  1️⃣ Recording initial sync attempt...")
    status = SyncStatus(
        entity_id=entity_id,
        last_attempt=now,
        last_success=None  # First attempt, no success yet
    )
    session.add(status)
    session.commit()
    
    # Add some samples
    print("  2️⃣ Adding samples...")
    samples = [
        Sample(entity_id=entity_id, timestamp=now, value=20.0, unit="°C"),
        Sample(entity_id=entity_id, timestamp=now + timedelta(minutes=5), value=21.0, unit="°C"),
    ]
    session.add_all(samples)
    session.commit()
    
    # Mark sync as successful
    print("  3️⃣ Marking sync successful...")
    status.last_success = now + timedelta(seconds=30)
    session.commit()
    
    # Verify sync status
    retrieved = session.query(SyncStatus).filter_by(entity_id=entity_id).first()
    assert retrieved is not None
    assert retrieved.last_success is not None
    
    print("✅ Sync status workflow completed!")
    
    session.close()


@pytest.mark.integration
def test_multi_category_processing(mariadb_engine, clean_database):
    """
    Test processing multiple sensor categories simultaneously.
    """
    session = Session(mariadb_engine)
    
    print("\n🔄 Testing multi-category processing...")
    
    # Setup multiple categories
    categories = [
        ("outdoor_temp", "sensor.outdoor", "°C"),
        ("indoor_temp", "sensor.indoor", "°C"),
        ("wind_speed", "sensor.wind", "m/s"),
        ("humidity", "sensor.humidity", "%"),
    ]
    
    print("  1️⃣ Setting up sensor mappings...")
    for cat, entity, unit in categories:
        mapping = SensorMapping(
            category=cat,
            entity_id=entity,
            is_active=True,
            priority=1
        )
        session.add(mapping)
    session.commit()
    
    # Insert samples for all categories
    print("  2️⃣ Inserting samples for all categories...")
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    samples = []
    
    for i in range(12):  # 1 hour of data
        timestamp = base_time + timedelta(minutes=i*5)
        for cat, entity, unit in categories:
            samples.append(Sample(
                entity_id=entity,
                timestamp=timestamp,
                value=20.0 + i,
                unit=unit
            ))
    
    session.add_all(samples)
    session.commit()
    print(f"    ✅ Inserted {len(samples)} samples")
    
    # Resample all categories
    print("  3️⃣ Resampling all categories...")
    from db.resample import resample_category
    
    for i in range(12):
        slot_start = base_time + timedelta(minutes=i*5)
        slot_end = slot_start + timedelta(minutes=5)
        for cat, _, _ in categories:
            resample_category(cat, slot_start, slot_end)
    
    resampled_count = session.query(ResampledSample).count()
    print(f"    ✅ Created {resampled_count} resampled entries")
    
    # Calculate stats for all categories
    print("  4️⃣ Calculating statistics...")
    from db.calculate_feature_stats import calculate_feature_stats_for_slot
    
    target_slot = base_time + timedelta(minutes=11*5)
    calculate_feature_stats_for_slot(target_slot)
    
    stats_count = session.query(FeatureStatistic).filter_by(slot_start=target_slot).count()
    print(f"    ✅ Calculated {stats_count} statistics")
    
    # Verify all categories were processed
    for cat, _, _ in categories:
        cat_resampled = session.query(ResampledSample).filter_by(category=cat).count()
        print(f"    📊 {cat}: {cat_resampled} resampled entries")
        assert cat_resampled > 0
    
    print("\n✅ Multi-category processing completed!")
    
    session.close()


@pytest.mark.integration
def test_data_integrity_constraints(mariadb_engine, clean_database):
    """
    Test that database constraints maintain data integrity.
    """
    from sqlalchemy.exc import IntegrityError
    
    session = Session(mariadb_engine)
    
    print("\n🔄 Testing data integrity constraints...")
    
    # Test unique constraint on samples
    print("  1️⃣ Testing unique constraint on samples...")
    now = datetime.utcnow()
    
    sample1 = Sample(entity_id="sensor.test", timestamp=now, value=20.0, unit="°C")
    session.add(sample1)
    session.commit()
    
    # Try to insert duplicate
    sample2 = Sample(entity_id="sensor.test", timestamp=now, value=30.0, unit="°C")
    session.add(sample2)
    
    with pytest.raises(IntegrityError):
        session.commit()
    
    session.rollback()
    print("    ✅ Unique constraint enforced")
    
    # Test unique constraint on resampled samples
    print("  2️⃣ Testing unique constraint on resampled samples...")
    slot_start = datetime(2024, 1, 1, 12, 0, 0)
    
    resampled1 = ResampledSample(
        slot_start=slot_start,
        category="outdoor_temp",
        value=20.0,
        unit="°C",
        is_derived=False
    )
    session.add(resampled1)
    session.commit()
    
    # Try to insert duplicate
    resampled2 = ResampledSample(
        slot_start=slot_start,
        category="outdoor_temp",
        value=25.0,
        unit="°C",
        is_derived=False
    )
    session.add(resampled2)
    
    with pytest.raises(IntegrityError):
        session.commit()
    
    session.rollback()
    print("    ✅ Resampled unique constraint enforced")
    
    print("\n✅ Data integrity tests passed!")
    
    session.close()


@pytest.mark.integration
def test_large_dataset_performance(mariadb_engine, clean_database):
    """
    Test system performance with a large dataset.
    """
    import time
    session = Session(mariadb_engine)
    
    print("\n🔄 Testing large dataset performance...")
    
    # Setup
    mapping = SensorMapping(
        category="outdoor_temp",
        entity_id="sensor.outdoor",
        is_active=True,
        priority=1
    )
    session.add(mapping)
    session.commit()
    
    # Insert a large number of samples (7 days = ~2000 samples)
    print("  1️⃣ Inserting 2000+ samples...")
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    samples = []
    
    start = time.time()
    for i in range(2016):  # 7 days * 288 samples/day
        timestamp = base_time + timedelta(minutes=i*5)
        samples.append(Sample(
            entity_id="sensor.outdoor",
            timestamp=timestamp,
            value=10.0 + (i % 100) * 0.1,
            unit="°C"
        ))
    
    session.add_all(samples)
    session.commit()
    insert_duration = time.time() - start
    print(f"    ⏱️ Insert took {insert_duration:.2f}s")
    
    # Query performance test
    print("  2️⃣ Testing query performance...")
    start = time.time()
    count = session.query(Sample).filter_by(entity_id="sensor.outdoor").count()
    query_duration = time.time() - start
    print(f"    ⏱️ Count query took {query_duration:.3f}s")
    
    assert count == 2016
    assert query_duration < 1.0
    
    print("\n✅ Performance tests passed!")
    
    session.close()
