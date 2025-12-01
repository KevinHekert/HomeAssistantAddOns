"""
Tests for sensor configuration management.
"""

import pytest
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import Base, SensorMapping
from db.sensor_config import (
    SENSOR_CATEGORIES,
    DEFAULT_ENTITIES,
    get_configured_sensors,
    sync_sensor_mappings,
)
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
def clean_env(monkeypatch):
    """Clear all sensor-related environment variables."""
    for env_var in SENSOR_CATEGORIES.values():
        monkeypatch.delenv(env_var, raising=False)
    yield


class TestGetConfiguredSensors:
    """Test the get_configured_sensors function."""

    def test_defaults_used_when_no_env_vars(self, clean_env):
        """Default entity IDs are used when no environment variables are set."""
        result = get_configured_sensors()
        assert result == DEFAULT_ENTITIES

    def test_env_var_overrides_default(self, clean_env, monkeypatch):
        """Environment variable overrides default value."""
        monkeypatch.setenv("WIND_ENTITY_ID", "sensor.custom_wind")
        result = get_configured_sensors()
        assert result["wind"] == "sensor.custom_wind"
        # Other defaults should still be present
        assert result["outdoor_temp"] == DEFAULT_ENTITIES["outdoor_temp"]

    def test_all_env_vars_used(self, clean_env, monkeypatch):
        """All environment variables are used when set."""
        custom_sensors = {
            "WIND_ENTITY_ID": "sensor.custom_wind",
            "OUTDOOR_TEMP_ENTITY_ID": "sensor.custom_temp",
            "FLOW_TEMP_ENTITY_ID": "sensor.custom_flow",
            "RETURN_TEMP_ENTITY_ID": "sensor.custom_return",
            "HUMIDITY_ENTITY_ID": "sensor.custom_humidity",
            "PRESSURE_ENTITY_ID": "sensor.custom_pressure",
            "HP_KWH_TOTAL_ENTITY_ID": "sensor.custom_kwh",
            "DHW_TEMP_ENTITY_ID": "sensor.custom_dhw",
        }
        for env_var, value in custom_sensors.items():
            monkeypatch.setenv(env_var, value)

        result = get_configured_sensors()
        assert result["wind"] == "sensor.custom_wind"
        assert result["outdoor_temp"] == "sensor.custom_temp"
        assert result["flow_temp"] == "sensor.custom_flow"
        assert result["return_temp"] == "sensor.custom_return"
        assert result["humidity"] == "sensor.custom_humidity"
        assert result["pressure"] == "sensor.custom_pressure"
        assert result["hp_kwh_total"] == "sensor.custom_kwh"
        assert result["dhw_temp"] == "sensor.custom_dhw"


class TestSyncSensorMappings:
    """Test the sync_sensor_mappings function."""

    def test_creates_mappings_from_defaults(self, patch_engine, clean_env):
        """Creates sensor mappings from default values."""
        sync_sensor_mappings()

        with Session(patch_engine) as session:
            mappings = session.query(SensorMapping).all()
            assert len(mappings) == len(DEFAULT_ENTITIES)

            for mapping in mappings:
                assert mapping.is_active is True
                assert mapping.priority == 1
                assert mapping.entity_id == DEFAULT_ENTITIES[mapping.category]

    def test_creates_mappings_from_env_vars(self, patch_engine, clean_env, monkeypatch):
        """Creates sensor mappings from environment variables."""
        monkeypatch.setenv("WIND_ENTITY_ID", "sensor.custom_wind")

        sync_sensor_mappings()

        with Session(patch_engine) as session:
            wind_mapping = (
                session.query(SensorMapping)
                .filter(SensorMapping.category == "wind")
                .first()
            )
            assert wind_mapping is not None
            assert wind_mapping.entity_id == "sensor.custom_wind"
            assert wind_mapping.is_active is True

    def test_reactivates_inactive_mapping(self, patch_engine, clean_env):
        """Reactivates an existing inactive mapping."""
        # Create an inactive mapping
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="wind",
                    entity_id=DEFAULT_ENTITIES["wind"],
                    is_active=False,
                    priority=1,
                )
            )
            session.commit()

        sync_sensor_mappings()

        with Session(patch_engine) as session:
            wind_mapping = (
                session.query(SensorMapping)
                .filter(SensorMapping.category == "wind")
                .first()
            )
            assert wind_mapping is not None
            assert wind_mapping.is_active is True

    def test_idempotent_sync(self, patch_engine, clean_env):
        """Running sync multiple times doesn't create duplicates."""
        sync_sensor_mappings()
        sync_sensor_mappings()
        sync_sensor_mappings()

        with Session(patch_engine) as session:
            mappings = session.query(SensorMapping).all()
            # Should still only have one mapping per category
            assert len(mappings) == len(DEFAULT_ENTITIES)

    def test_preserves_existing_mappings(self, patch_engine, clean_env):
        """Existing mappings with different entity_id are preserved."""
        # Create a custom mapping
        with Session(patch_engine) as session:
            session.add(
                SensorMapping(
                    category="wind",
                    entity_id="sensor.manual_wind",
                    is_active=True,
                    priority=2,
                )
            )
            session.commit()

        sync_sensor_mappings()

        with Session(patch_engine) as session:
            wind_mappings = (
                session.query(SensorMapping)
                .filter(SensorMapping.category == "wind")
                .all()
            )
            # Should have both the manual and the default mapping
            assert len(wind_mappings) == 2
            entity_ids = {m.entity_id for m in wind_mappings}
            assert "sensor.manual_wind" in entity_ids
            assert DEFAULT_ENTITIES["wind"] in entity_ids
