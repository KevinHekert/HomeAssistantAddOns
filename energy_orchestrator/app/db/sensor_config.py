"""
Sensor configuration management.

This module handles loading sensor mappings from configuration and populating
the sensor_mappings table in the database.

The primary configuration source is now the sensor_category_config module
which stores settings in /data/sensor_category_config.json.
Environment variables are used for initial migration on first run.
"""

import logging

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db import SensorMapping
from db.core import engine
from db.sensor_category_config import get_sensor_category_config

_Logger = logging.getLogger(__name__)


def sync_sensor_mappings() -> None:
    """
    Synchronize sensor mappings from configuration to database.

    This function:
    1. Reads sensor configurations from the sensor category config
    2. For each enabled sensor, ensures a mapping exists in the database
    3. Updates existing mappings if the entity_id has changed
    4. Does not deactivate mappings not in config (allows manual additions)
    
    Note: The sensor_category_config module automatically migrates from
    environment variables (config.yaml) on first run.
    """
    # Get configuration (triggers migration from env vars if needed)
    config = get_sensor_category_config()
    enabled_sensors = config.get_enabled_sensors()

    if not enabled_sensors:
        _Logger.warning("No sensor configurations found. Check sensor configuration.")
        return

    try:
        with Session(engine) as session:
            for sensor_config in enabled_sensors:
                category = sensor_config.category_name
                entity_id = sensor_config.entity_id
                
                if not entity_id:
                    _Logger.debug("Skipping %s: no entity_id configured", category)
                    continue
                
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
                "Sensor mappings synchronized: %d sensors enabled.",
                len(enabled_sensors),
            )

    except SQLAlchemyError as e:
        _Logger.error("Error synchronizing sensor mappings: %s", e)
