"""
Shared pytest configuration and fixtures for all tests.

This file provides test database setup using SQLite in-memory databases
instead of the production MariaDB database.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# Import db.core first to patch before any other module uses it
import db.core
from db import Base


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    Automatically patch the database engine for all tests.
    
    This fixture runs once per test session and replaces the production
    MariaDB connection with an in-memory SQLite database.
    """
    # Import all db modules that need patching
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
    
    # Create an in-memory SQLite database engine
    test_engine = create_engine("sqlite:///:memory:", echo=False)
    
    # Create all tables
    Base.metadata.create_all(test_engine)
    
    # Patch the engine in db.core module so all imports use the test database
    db.core.engine = test_engine
    
    # Patch engine in all db modules
    db.samples.engine = test_engine
    db.resample.engine = test_engine
    db.feature_stats.engine = test_engine
    db.optimizer_storage.engine = test_engine
    db.prediction_storage.engine = test_engine
    db.sensor_config.engine = test_engine
    db.sensor_category_config.engine = test_engine
    db.optimizer_config.engine = test_engine
    db.sync_config.engine = test_engine
    db.sync_state.engine = test_engine
    db.virtual_sensors.engine = test_engine
    
    yield test_engine
    
    # Cleanup after all tests
    Base.metadata.drop_all(test_engine)
    test_engine.dispose()


@pytest.fixture
def test_session(setup_test_database):
    """
    Provide a database session for tests that need direct database access.
    
    Each test gets a fresh session that is rolled back after the test.
    """
    session = Session(setup_test_database)
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def clean_database(setup_test_database):
    """
    Clean all data from the database between tests.
    
    Use this fixture for tests that need a completely clean database.
    """
    # Clear all tables
    for table in reversed(Base.metadata.sorted_tables):
        with setup_test_database.connect() as conn:
            conn.execute(table.delete())
            conn.commit()
    
    yield setup_test_database


@pytest.fixture(autouse=True)
def reset_resample_state():
    """
    Reset the global resampling state between tests.
    
    This ensures tests don't interfere with each other when testing
    the background resampling functionality.
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
