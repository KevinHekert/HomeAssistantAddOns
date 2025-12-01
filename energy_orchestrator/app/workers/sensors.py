import os
import logging
import time
import threading

from db.core import test_db_connection, init_db_schema
from db.samples import get_latest_sample_timestamp
from db.sync_state import get_sync_status
from ha.ha_api import sync_history_for_entity

_Logger = logging.getLogger(__name__)

# Let op: entity_id’s mogen geen spaties hebben – pas deze defaults aan naar jouw echte IDs
SENSOR_ENTITIES = [
    os.environ.get("WIND_ENTITY_ID", "sensor.knmi_windsnelheid"),
    os.environ.get("OUTDOOR_TEMP_ENTITY_ID", "sensor.smile_outdoor_temperature"),
    os.environ.get("FLOW_TEMP_ENTITY_ID", "sensor.opentherm_water_temperature"),
    os.environ.get("RETURN_TEMP_ENTITY_ID", "sensor.opentherm_return_temperature"),
    os.environ.get("HUMIDITY_ENTITY_ID", "sensor.knmi_luchtvochtigheid"),
    os.environ.get("PRESSURE_ENTITY_ID", "sensor.knmi_luchtdruk"),
    os.environ.get("HP_KWH_TOTAL_ENTITY_ID", "sensor.extra_total"),
    os.environ.get("DHW_TEMP_ENTITY_ID", "sensor.opentherm_dhw_temperature"),
    os.environ.get("INDOOR_TEMP_ENTITY_ID", "sensor.anna_temperature"),
    os.environ.get("TARGET_TEMP_ENTITY_ID", "sensor.anna_setpoint"),
    os.environ.get("DHW_ACTIVE_ENTITY_ID", "binary_sensor.dhw_active"),
]

_sensor_worker_started = False


def _sync_entity(entity_id: str) -> None:
    """Voer één sync-cyclus uit voor een enkele entity."""
    if not entity_id:
        return

    latest_ts = get_latest_sample_timestamp(entity_id)
    status = get_sync_status(entity_id)

    if latest_ts is not None:
        effective_since = latest_ts
    elif status is not None and status.last_attempt is not None:
        effective_since = status.last_attempt
    else:
        effective_since = None

    _Logger.info(
        "Effective since voor %s vóór sync: %s",
        entity_id,
        effective_since,
    )

    sync_history_for_entity(entity_id, effective_since)
    time.sleep(5)


def sensor_logging_worker():
    """Achtergrondthread die periodiek alle sensoren sync't uit HA-history."""
    _Logger.info("Generieke sensor logging worker gestart.")
    while True:
        try:
            test_db_connection()
            init_db_schema()

            for entity_id in SENSOR_ENTITIES:
                try:
                    _sync_entity(entity_id)
                except Exception as e:
                    _Logger.error(
                        "Onverwachte fout tijdens sync voor %s: %s",
                        entity_id,
                        e,
                    )

        except Exception as e:
            _Logger.error("Onverwachte fout in sensor logging worker-loop: %s", e)

        # TODO: voor productie naar 300s; nu 10 voor sneller testen
        time.sleep(300)


def start_sensor_logging_worker():
    global _sensor_worker_started
    if _sensor_worker_started:
        return

    thread = threading.Thread(target=sensor_logging_worker, daemon=True)
    thread.start()
    _sensor_worker_started = True
    _Logger.info("Sensor logging worker thread gestart.")
