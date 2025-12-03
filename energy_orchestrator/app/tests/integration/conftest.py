"""
Integration test configuration for Energy Orchestrator with MariaDB.

This module provides pytest fixtures and configuration for integration testing
with a real MariaDB database instead of SQLite in-memory database.
"""

import os
import time
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from db import Base
import db.core


@pytest.fixture(scope="session")
def mariadb_engine():
    """
    Create a MariaDB engine for integration testing.
    
    This fixture connects to the MariaDB test database running in Docker.
    It waits for the database to be ready before proceeding.
    
    Environment variables:
    - DB_HOST: MariaDB hostname (default: localhost)
    - DB_PORT: MariaDB port (default: 3307)
    - DB_USER: Database user (default: energy_orchestrator)
    - DB_PASSWORD: Database password (default: test_password)
    - DB_NAME: Database name (default: energy_orchestrator_test)
    """
    # Get database connection details from environment
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "3307")
    db_user = os.environ.get("DB_USER", "energy_orchestrator")
    db_password = os.environ.get("DB_PASSWORD", "test_password")
    db_name = os.environ.get("DB_NAME", "energy_orchestrator_test")
    
    # Build connection URL
    db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    print(f"\n🔗 Connecting to MariaDB at {db_host}:{db_port}/{db_name}")
    
    # Wait for database to be ready (max 30 seconds)
    max_retries = 30
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            engine = create_engine(db_url, future=True, echo=False)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"✅ Connected to MariaDB successfully")
            break
        except OperationalError as e:
            if attempt < max_retries - 1:
                print(f"⏳ Waiting for MariaDB... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to connect to MariaDB after {max_retries} attempts: {e}")
    
    # Create all tables
    print("📋 Creating database schema...")
    Base.metadata.create_all(engine)
    print("✅ Database schema created")
    
    # Import all db modules before patching to avoid Python's UnboundLocalError
    # (assignments to db.* attributes make Python think 'db' is local)
    import db.samples
    import db.resample
    import db.feature_stats
    import db.optimizer_storage
    import db.prediction_storage
    import db.sensor_config
    import db.sensor_category_config
    import db.optimizer_config
    import db.sync_config
    import db.sync_state
    import db.virtual_sensors
    
    # Store references to modules to avoid UnboundLocalError with db namespace
    db_core = db.core
    db_samples = db.samples
    db_resample = db.resample
    db_feature_stats = db.feature_stats
    db_optimizer_storage = db.optimizer_storage
    db_prediction_storage = db.prediction_storage
    db_sensor_config = db.sensor_config
    db_sensor_category_config = db.sensor_category_config
    db_optimizer_config = db.optimizer_config
    db_sync_config = db.sync_config
    db_sync_state = db.sync_state
    db_virtual_sensors = db.virtual_sensors
    
    # Save original engine from db.core for restoration later
    original_engine = db_core.engine
    
    # Patch db.core and all db module engines to use the test engine
    db_core.engine = engine
    db_samples.engine = engine
    db_resample.engine = engine
    db_feature_stats.engine = engine
    db_optimizer_storage.engine = engine
    db_prediction_storage.engine = engine
    db_sensor_config.engine = engine
    db_sensor_category_config.engine = engine
    db_optimizer_config.engine = engine
    db_sync_config.engine = engine
    db_sync_state.engine = engine
    db_virtual_sensors.engine = engine
    
    yield engine
    
    # Cleanup: drop all tables and restore original engine
    print("\n🧹 Cleaning up database...")
    Base.metadata.drop_all(engine)
    engine.dispose()
    db_core.engine = original_engine
    print("✅ Cleanup complete")


@pytest.fixture
def integration_session(mariadb_engine):
    """
    Provide a database session for integration tests.
    
    Each test gets a fresh session with automatic rollback.
    """
    session = Session(mariadb_engine)
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def clean_database(mariadb_engine):
    """
    Clean all data from the database between tests.
    
    This fixture truncates all tables to ensure tests start with a clean slate.
    Use this for tests that need complete isolation.
    """
    # Disable foreign key checks temporarily for truncation
    with mariadb_engine.connect() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        
        # Truncate all tables
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f"TRUNCATE TABLE {table.name}"))
        
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        conn.commit()
    
    yield mariadb_engine


@pytest.fixture
def sample_test_data(integration_session):
    """
    Insert sample test data for integration tests.
    
    This fixture provides a standard set of test data including:
    - Sensor configurations
    - Sample measurements
    - Resampled data
    
    Returns the session with test data loaded.
    """
    from datetime import datetime, timedelta
    from db.models import Sample, SensorMapping, ResampledSample
    
    # Create sensor mappings
    sensors = [
        SensorMapping(
            category="outdoor_temp",
            entity_id="sensor.smile_outdoor_temperature",
            is_active=True,
            priority=1
        ),
        SensorMapping(
            category="indoor_temp",
            entity_id="sensor.anna_temperature",
            is_active=True,
            priority=1
        ),
        SensorMapping(
            category="wind_speed",
            entity_id="sensor.knmi_windsnelheid",
            is_active=True,
            priority=1
        ),
    ]
    integration_session.add_all(sensors)
    
    # Create sample data for the last 24 hours
    now = datetime.utcnow()
    samples = []
    
    for i in range(288):  # 24 hours * 60 minutes / 5 minutes = 288 samples
        timestamp = now - timedelta(minutes=i*5)
        
        samples.extend([
            Sample(
                entity_id="sensor.smile_outdoor_temperature",
                timestamp=timestamp,
                value=8.0 + (i % 24) * 0.5,  # Varies between 8-20°C
                unit="°C"
            ),
            Sample(
                entity_id="sensor.anna_temperature",
                timestamp=timestamp,
                value=21.0 + (i % 10) * 0.1,  # Varies between 21-22°C
                unit="°C"
            ),
            Sample(
                entity_id="sensor.knmi_windsnelheid",
                timestamp=timestamp,
                value=3.0 + (i % 15) * 0.3,  # Varies between 3-7.5 m/s
                unit="m/s"
            ),
        ])
    
    integration_session.add_all(samples)
    integration_session.commit()
    
    print(f"✅ Inserted {len(samples)} test samples")
    
    yield integration_session


@pytest.fixture(autouse=True)
def reset_app_state():
    """
    Reset global application state between integration tests.
    
    This ensures integration tests don't interfere with each other.
    """
    import app
    
    # Reset global state
    app._resample_running = False
    app._resample_progress = None
    app._resample_thread = None
    
    yield
    
    # Clean up after test
    app._resample_running = False
    app._resample_progress = None
    app._resample_thread = None
