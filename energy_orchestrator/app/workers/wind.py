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

# Later komt dit uit config/integratie; nu gewoon env of default
WIND_ENTITY_ID = os.environ.get("WIND_ENTITY_ID", "sensor.knmi_windsnelheid")

_wind_logger_started = False


def wind_logging_worker():
    """Achtergrondthread die periodiek samples sync't uit HA-history."""
    _Logger.info("Wind logging worker gestart voor %s.", WIND_ENTITY_ID)
    while True:
        try:
            test_db_connection()
            init_db_schema()

            latest_ts = get_latest_sample_timestamp(WIND_ENTITY_ID)
            status = get_sync_status(WIND_ENTITY_ID)

            if latest_ts is not None:
                effective_since = latest_ts
            elif status is not None and status.last_attempt is not None:
                effective_since = status.last_attempt
            else:
                effective_since = None

            _Logger.info(
                "Effective since voor %s vóór sync: %s",
                WIND_ENTITY_ID,
                effective_since,
            )

            sync_history_for_entity(WIND_ENTITY_ID, effective_since)

        except Exception as e:
            _Logger.error("Onverwachte fout in wind logging worker: %s", e)
        # Elke 10 seconden opnieuw proberen
        time.sleep(1)


def start_wind_logging_worker():
    global _wind_logger_started
    if _wind_logger_started:
        return

    thread = threading.Thread(target=wind_logging_worker, daemon=True)
    thread.start()
    _wind_logger_started = True
    _Logger.info("Wind logging worker thread gestart.")
