import os
import logging
import time

from datetime import datetime, timezone

from db.core import test_db_connection, init_db_schema
from db.samples import get_latest_sample_timestamp
from db.sync_state import get_sync_status

from ha.ha_api import sync_history_for_entity
import threading

_Logger = logging.getLogger(__name__)

# Later uit config/integratie; nu env of default
OUTDOOR_TEMP_ENTITY_ID = os.environ.get(
    "OUTDOOR_TEMP_ENTITY_ID",
    "sensor.smile_outdoor_temperature",  # pas aan als jouw entity anders heet
)

_temp_logger_started = False


def temperature_logging_worker():
    """Achtergrondthread die periodiek buitentemperatuur samples sync't uit HA-history."""
    _Logger.info(
        "Temperatuur logging worker gestart voor %s.", OUTDOOR_TEMP_ENTITY_ID
    )
    while True:
        try:
            test_db_connection()
            init_db_schema()

            latest_ts = get_latest_sample_timestamp(OUTDOOR_TEMP_ENTITY_ID)
            status = get_sync_status(OUTDOOR_TEMP_ENTITY_ID)

            if latest_ts is not None:
                effective_since = latest_ts
            elif status is not None and status.last_attempt is not None:
                effective_since = status.last_attempt
            else:
                effective_since = None

            _Logger.info(
                "Effective since voor %s vóór sync: %s",
                OUTDOOR_TEMP_ENTITY_ID,
                effective_since,
            )

            sync_history_for_entity(OUTDOOR_TEMP_ENTITY_ID, effective_since)

        except Exception as e:
            _Logger.error(
                "Onverwachte fout in temperatuur logging worker: %s", e
            )

        # Later naar 300s of zo; nu kort voor testen
        time.sleep(10)


def start_temperature_logging_worker():
    global _temp_logger_started
    if _temp_logger_started:
        return

    thread = threading.Thread(target=temperature_logging_worker, daemon=True)
    thread.start()
    _temp_logger_started = True
    _Logger.info("Temperatuur logging worker thread gestart.")
