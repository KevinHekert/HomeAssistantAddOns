"""
Tests for sensor cards API endpoint.

This test validates that sensor cards properly display all configured
time-based statistics (avg_1h, avg_6h, avg_24h, avg_7d) as individual
checkboxes, not just the base sensor value.
"""

import pytest
from flask import Flask
from sqlalchemy import create_engine, text
from db import Base
from db.sensor_category_config import get_sensor_category_config
from db.feature_stats import get_feature_stats_config, StatType
from ml.feature_config import get_feature_config, reload_feature_config
from app import app as flask_app
import db.core as core_module


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def patch_engine(test_engine, monkeypatch):
    """Patch the engine in the core module."""
    monkeypatch.setattr(core_module, "engine", test_engine)
    return test_engine


@pytest.fixture
def app():
    """Create Flask app for testing."""
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


@pytest.fixture(autouse=True)
def setup_database(patch_engine):
    """Initialize database schema before each test using SQLite."""
    # Schema is already created by test_engine fixture
    yield
    # Cleanup after test - SQLite doesn't support DELETE FROM but we can drop and recreate
    # For SQLite in-memory, this isn't strictly necessary but good practice
    pass


def test_sensor_cards_show_configured_statistics(client):
    """
    Test that sensor cards include all configured time-based statistics.
    
    This test validates the fix for the issue where humidity and other sensors
    only showed one checkbox (the sensor itself) instead of showing checkboxes
    for each configured time-based statistic (avg_1h, avg_6h, avg_24h, etc.).
    """
    # Get feature stats configuration
    feature_stats_conf = get_feature_stats_config()
    
    # Enable some statistics for humidity sensor
    feature_stats_conf.set_stat_enabled("humidity", StatType.AVG_1H, True)
    feature_stats_conf.set_stat_enabled("humidity", StatType.AVG_6H, True)
    feature_stats_conf.set_stat_enabled("humidity", StatType.AVG_24H, True)
    feature_stats_conf.save()
    
    # Get sensor cards
    response = client.get("/api/features/sensor_cards")
    assert response.status_code == 200
    
    data = response.get_json()
    assert data["status"] == "success"
    assert "sensor_cards" in data
    
    # Find the humidity card
    humidity_card = None
    for card in data["sensor_cards"]:
        if card["sensor_name"] == "humidity":
            humidity_card = card
            break
    
    assert humidity_card is not None, "Humidity sensor card not found"
    
    # Check that the humidity card has features
    features = humidity_card["features"]
    assert len(features) > 0, "Humidity card should have features"
    
    # Check that we have the base humidity feature
    feature_names = [f["name"] for f in features]
    assert "humidity" in feature_names, "Base humidity feature should be present"
    
    # Check that we have the configured time-based statistics
    assert "humidity_avg_1h" in feature_names, "humidity_avg_1h should be present"
    assert "humidity_avg_6h" in feature_names, "humidity_avg_6h should be present"
    assert "humidity_avg_24h" in feature_names, "humidity_avg_24h should be present"
    
    # Verify that each feature has the required fields
    for feature in features:
        assert "name" in feature
        assert "display_name" in feature
        assert "description" in feature
        assert "unit" in feature
        assert "time_window" in feature
        assert "is_core" in feature
        assert "enabled" in feature
    
    print(f"✓ Humidity card has {len(features)} features: {feature_names}")


def test_sensor_cards_all_sensors_show_statistics(client):
    """
    Test that all sensors properly show their configured statistics.
    
    This validates that the fix works for all sensors, not just humidity.
    """
    # Get feature stats configuration
    feature_stats_conf = get_feature_stats_config()
    
    # Configure statistics for multiple sensors
    test_sensors = ["outdoor_temp", "indoor_temp", "pressure"]
    for sensor_name in test_sensors:
        feature_stats_conf.set_stat_enabled(sensor_name, StatType.AVG_1H, True)
        feature_stats_conf.set_stat_enabled(sensor_name, StatType.AVG_6H, True)
        feature_stats_conf.set_stat_enabled(sensor_name, StatType.AVG_24H, True)
    feature_stats_conf.save()
    
    # Get sensor cards
    response = client.get("/api/features/sensor_cards")
    assert response.status_code == 200
    
    data = response.get_json()
    assert data["status"] == "success"
    
    # Check each test sensor
    for sensor_name in test_sensors:
        sensor_card = None
        for card in data["sensor_cards"]:
            if card["sensor_name"] == sensor_name:
                sensor_card = card
                break
        
        if sensor_card is None:
            # Sensor might not be configured, skip
            continue
        
        features = sensor_card["features"]
        feature_names = [f["name"] for f in features]
        
        # Verify base sensor and time-based statistics
        assert sensor_name in feature_names, f"Base {sensor_name} feature should be present"
        assert f"{sensor_name}_avg_1h" in feature_names, f"{sensor_name}_avg_1h should be present"
        assert f"{sensor_name}_avg_6h" in feature_names, f"{sensor_name}_avg_6h should be present"
        assert f"{sensor_name}_avg_24h" in feature_names, f"{sensor_name}_avg_24h should be present"
        
        print(f"✓ {sensor_name} card has {len(features)} features: {feature_names}")


def test_sensor_cards_without_statistics_still_work(client):
    """
    Test that sensors without configured statistics still display correctly.
    
    This ensures we didn't break existing functionality.
    """
    # Get feature stats configuration and disable all stats for a sensor
    feature_stats_conf = get_feature_stats_config()
    feature_stats_conf.set_stat_enabled("wind", StatType.AVG_1H, False)
    feature_stats_conf.set_stat_enabled("wind", StatType.AVG_6H, False)
    feature_stats_conf.set_stat_enabled("wind", StatType.AVG_24H, False)
    feature_stats_conf.set_stat_enabled("wind", StatType.AVG_7D, False)
    feature_stats_conf.save()
    
    # Get sensor cards
    response = client.get("/api/features/sensor_cards")
    assert response.status_code == 200
    
    data = response.get_json()
    assert data["status"] == "success"
    
    # Find wind card
    wind_card = None
    for card in data["sensor_cards"]:
        if card["sensor_name"] == "wind":
            wind_card = card
            break
    
    if wind_card is not None:
        features = wind_card["features"]
        feature_names = [f["name"] for f in features]
        
        # Should still have the base wind feature
        assert "wind" in feature_names, "Base wind feature should be present"
        
        print(f"✓ Wind card (no stats) has {len(features)} features: {feature_names}")


def test_sensor_cards_display_names(client):
    """
    Test that feature display names are formatted correctly.
    """
    # Get feature stats configuration
    feature_stats_conf = get_feature_stats_config()
    feature_stats_conf.set_stat_enabled("outdoor_temp", StatType.AVG_1H, True)
    feature_stats_conf.set_stat_enabled("outdoor_temp", StatType.AVG_6H, True)
    feature_stats_conf.set_stat_enabled("outdoor_temp", StatType.AVG_24H, True)
    feature_stats_conf.set_stat_enabled("outdoor_temp", StatType.AVG_7D, True)
    feature_stats_conf.save()
    
    # Get sensor cards
    response = client.get("/api/features/sensor_cards")
    assert response.status_code == 200
    
    data = response.get_json()
    
    # Find outdoor_temp card
    outdoor_temp_card = None
    for card in data["sensor_cards"]:
        if card["sensor_name"] == "outdoor_temp":
            outdoor_temp_card = card
            break
    
    assert outdoor_temp_card is not None
    
    # Check display names
    for feature in outdoor_temp_card["features"]:
        name = feature["name"]
        display_name = feature["display_name"]
        
        if name == "outdoor_temp":
            assert display_name == "Current Value"
        elif "_avg_1h" in name:
            assert "1H" in display_name.upper()
        elif "_avg_6h" in name:
            assert "6H" in display_name.upper()
        elif "_avg_24h" in name:
            assert "24H" in display_name.upper()
        elif "_avg_7d" in name:
            assert "7D" in display_name.upper()
        
        print(f"  {name} -> {display_name}")
