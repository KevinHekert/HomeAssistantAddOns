"""
Tests for sensor configuration management.

Tests the sync_sensor_mappings function which uses the sensor_category_config module.
"""

import pytest
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import Base, SensorMapping
from db.sensor_config import sync_sensor_mappings
from db.sensor_category_config import SensorConfig, SensorCategoryConfiguration, CORE_SENSORS
import db.core as core_module
import db.sensor_config as config_module


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def patch_engine(test_engine, monkeypatch):
    """Patch the engine in both core and sensor_config modules."""
    monkeypatch.setattr(core_module, "engine", test_engine)
    monkeypatch.setattr(config_module, "engine", test_engine)
    return test_engine


@pytest.fixture
def mock_sensor_config():
    """Create a mock sensor category configuration."""
    config = SensorCategoryConfiguration()
    return config


class TestSyncSensorMappings:
    """Test the sync_sensor_mappings function."""

    def test_creates_mappings_from_config(self, patch_engine, tmp_path):
        """Creates sensor mappings from sensor category config."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_config.get_sensor_category_config") as mock_get_config:
            # Create a mock config with enabled sensors
            mock_config = MagicMock()
            mock_config.get_enabled_sensors.return_value = [
                SensorConfig(category_name="outdoor_temp", entity_id="sensor.outdoor_temp", enabled=True),
                SensorConfig(category_name="wind", entity_id="sensor.wind", enabled=True),
            ]
            mock_get_config.return_value = mock_config
            
            sync_sensor_mappings()

            with Session(patch_engine) as session:
                mappings = session.query(SensorMapping).all()
                assert len(mappings) == 2
                
                categories = {m.category for m in mappings}
                assert "outdoor_temp" in categories
                assert "wind" in categories

    def test_creates_mapping_with_correct_entity_id(self, patch_engine, tmp_path):
        """Mapping should have the correct entity_id from config."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_config.get_sensor_category_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.get_enabled_sensors.return_value = [
                SensorConfig(category_name="hp_kwh_total", entity_id="sensor.my_kwh", enabled=True),
            ]
            mock_get_config.return_value = mock_config
            
            sync_sensor_mappings()

            with Session(patch_engine) as session:
                mapping = session.query(SensorMapping).filter(
                    SensorMapping.category == "hp_kwh_total"
                ).first()
                
                assert mapping is not None
                assert mapping.entity_id == "sensor.my_kwh"
                assert mapping.is_active is True

    def test_reactivates_inactive_mapping(self, patch_engine, tmp_path):
        """Reactivates an existing inactive mapping."""
        # Create an inactive mapping
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="wind",
                    entity_id="sensor.wind",
                    is_active=False,
                    priority=1,
                )
            )
            session.commit()

        with patch("db.sensor_config.get_sensor_category_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.get_enabled_sensors.return_value = [
                SensorConfig(category_name="wind", entity_id="sensor.wind", enabled=True),
            ]
            mock_get_config.return_value = mock_config
            
            sync_sensor_mappings()

            with Session(patch_engine) as session:
                wind_mapping = (
                    session.query(SensorMapping)
                    .filter(SensorMapping.category == "wind")
                    .first()
                )
                assert wind_mapping is not None
                assert wind_mapping.is_active is True

    def test_idempotent_sync(self, patch_engine, tmp_path):
        """Running sync multiple times doesn't create duplicates."""
        with patch("db.sensor_config.get_sensor_category_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.get_enabled_sensors.return_value = [
                SensorConfig(category_name="outdoor_temp", entity_id="sensor.outdoor_temp", enabled=True),
            ]
            mock_get_config.return_value = mock_config
            
            sync_sensor_mappings()
            sync_sensor_mappings()
            sync_sensor_mappings()

            with Session(patch_engine) as session:
                mappings = session.query(SensorMapping).all()
                # Should still only have one mapping for outdoor_temp
                assert len(mappings) == 1

    def test_skips_sensors_without_entity_id(self, patch_engine, tmp_path):
        """Sensors without entity_id are skipped."""
        with patch("db.sensor_config.get_sensor_category_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.get_enabled_sensors.return_value = [
                SensorConfig(category_name="outdoor_temp", entity_id="sensor.outdoor_temp", enabled=True),
                SensorConfig(category_name="pressure", entity_id="", enabled=True),  # Empty entity_id
            ]
            mock_get_config.return_value = mock_config
            
            sync_sensor_mappings()

            with Session(patch_engine) as session:
                mappings = session.query(SensorMapping).all()
                # Only outdoor_temp should be created
                assert len(mappings) == 1
                assert mappings[0].category == "outdoor_temp"

    def test_handles_empty_sensor_list(self, patch_engine, tmp_path):
        """Handles case when no sensors are enabled."""
        with patch("db.sensor_config.get_sensor_category_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.get_enabled_sensors.return_value = []
            mock_get_config.return_value = mock_config
            
            # Should not raise an error
            sync_sensor_mappings()

            with Session(patch_engine) as session:
                mappings = session.query(SensorMapping).all()
                assert len(mappings) == 0

    def test_creates_all_core_sensors(self, patch_engine, tmp_path):
        """Creates mappings for all core sensors when using default config."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                # Use actual config which includes all core sensors
                sync_sensor_mappings()

                with Session(patch_engine) as session:
                    mappings = session.query(SensorMapping).all()
                    
                    # Should have at least all core sensors
                    assert len(mappings) >= len(CORE_SENSORS)
                    
                    categories = {m.category for m in mappings}
                    for core_sensor in CORE_SENSORS:
                        assert core_sensor.category_name in categories
