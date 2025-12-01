"""
Sensor configuration management.

This module handles loading sensor mappings from configuration (environment variables)
and populating the sensor_mappings table in the database.
"""

import os
import logging

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db import SensorMapping
from db.core import engine

_Logger = logging.getLogger(__name__)

# Category to environment variable mapping
SENSOR_CATEGORIES = {
    "wind": "WIND_ENTITY_ID",
    "outdoor_temp": "OUTDOOR_TEMP_ENTITY_ID",
    "flow_temp": "FLOW_TEMP_ENTITY_ID",
    "return_temp": "RETURN_TEMP_ENTITY_ID",
    "humidity": "HUMIDITY_ENTITY_ID",
    "pressure": "PRESSURE_ENTITY_ID",
    "hp_kwh_total": "HP_KWH_TOTAL_ENTITY_ID",
    "dhw_temp": "DHW_TEMP_ENTITY_ID",
    "indoor_temp": "INDOOR_TEMP_ENTITY_ID",
    "target_temp": "TARGET_TEMP_ENTITY_ID",
    "dhw_active": "DHW_ACTIVE_ENTITY_ID",
}

# Default fallback entity IDs (same as in workers/sensors.py)
DEFAULT_ENTITIES = {
    "wind": "sensor.knmi_windsnelheid",
    "outdoor_temp": "sensor.smile_outdoor_temperature",
    "flow_temp": "sensor.opentherm_water_temperature",
    "return_temp": "sensor.opentherm_return_temperature",
    "humidity": "sensor.knmi_luchtvochtigheid",
    "pressure": "sensor.knmi_luchtdruk",
    "hp_kwh_total": "sensor.extra_total",
    "dhw_temp": "sensor.opentherm_dhw_temperature",
    "indoor_temp": "sensor.anna_temperature",
    "target_temp": "sensor.anna_setpoint",
    "dhw_active": "binary_sensor.dhw_active",
}


def get_configured_sensors() -> dict[str, str]:
    """
    Get sensor entity IDs from environment variables.

    Returns:
        Dictionary mapping category names to entity IDs.
        Only includes categories where an entity ID is configured.
    """
    configured = {}
    for category, env_var in SENSOR_CATEGORIES.items():
        entity_id = os.environ.get(env_var, DEFAULT_ENTITIES.get(category))
        if entity_id:
            configured[category] = entity_id
    return configured


def sync_sensor_mappings() -> None:
    """
    Synchronize sensor mappings from configuration to database.

    This function:
    1. Reads sensor configurations from environment variables
    2. For each configured sensor, ensures a mapping exists in the database
    3. Updates existing mappings if the entity_id has changed
    4. Does not deactivate mappings not in config (allows manual additions)
    """
    configured = get_configured_sensors()

    if not configured:
        _Logger.warning("No sensor configurations found in environment.")
        return

    try:
        with Session(engine) as session:
            for category, entity_id in configured.items():
                # Check if mapping already exists for this category
                existing = (
                    session.query(SensorMapping)
                    .filter(
                        SensorMapping.category == category,
                        SensorMapping.entity_id == entity_id,
                    )
                    .first()
                )

                if existing:
                    # Ensure it's active
                    if not existing.is_active:
                        existing.is_active = True
                        _Logger.info(
                            "Reactivated sensor mapping: %s -> %s",
                            category,
                            entity_id,
                        )
                else:
                    # Create new mapping
                    mapping = SensorMapping(
                        category=category,
                        entity_id=entity_id,
                        is_active=True,
                        priority=1,
                    )
                    session.add(mapping)
                    _Logger.info(
                        "Created sensor mapping: %s -> %s",
                        category,
                        entity_id,
                    )

            session.commit()
            _Logger.info(
                "Sensor mappings synchronized: %d categories configured.",
                len(configured),
            )

    except SQLAlchemyError as e:
        _Logger.error("Error synchronizing sensor mappings: %s", e)
