import os
import logging
import time
import threading
from datetime import datetime, timedelta, timezone

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
    """
    Voer sync-cyclus(sen) uit voor een enkele entity.

    When no historic data is found and we haven't caught up to yesterday yet,
    loop faster (without the normal 5-second delay between iterations) until
    data is found or we reach today-1.
    """
    if not entity_id:
        return

    # Calculate yesterday once for the duration of this sync operation
    now_utc = datetime.now(timezone.utc)
    yesterday = now_utc - timedelta(days=1)

    # Safety limit to prevent infinite loops
    max_iterations = 200  # Should cover ~200 days of backfill max

    for iteration in range(max_iterations):
        latest_ts = get_latest_sample_timestamp(entity_id)
        status = get_sync_status(entity_id)

        if latest_ts is not None:
            effective_since = latest_ts
        elif status is not None and status.last_attempt is not None:
            effective_since = status.last_attempt
        else:
            effective_since = None

        _Logger.debug(
            "Effective since voor %s vóór sync: %s",
            entity_id,
            effective_since,
        )

        inserted = sync_history_for_entity(entity_id, effective_since)

        # Check if we should continue fast-forwarding:
        # - Only when no samples were inserted (inserted == 0)
        # - And no samples exist yet for this entity (latest_ts is None)
        # - And effective_since is before yesterday (today - 1 day)
        if inserted == 0 and latest_ts is None:
            # Normalize effective_since for comparison
            if effective_since is not None:
                if effective_since.tzinfo is None:
                    effective_since_aware = effective_since.replace(tzinfo=timezone.utc)
                else:
                    effective_since_aware = effective_since

                # If effective_since is before yesterday, fast-forward without delay
                if effective_since_aware < yesterday:
                    _Logger.debug(
                        "No data found for %s and not yet caught up to yesterday, fast-forwarding...",
                        entity_id,
                    )
                    continue
                else:
                    _Logger.debug(
                        "No data found for %s but caught up to yesterday, stopping fast-forward.",
                        entity_id,
                    )
            else:
                # effective_since is None means sync just started (first iteration)
                # After first sync, sync_status will be updated with last_attempt,
                # so continue to check progress
                _Logger.debug(
                    "First sync for %s (no prior data), continuing to check progress...",
                    entity_id,
                )
                continue

        # Normal case: wait before next entity
        time.sleep(5)
        break
    else:
        # Max iterations reached - log warning and continue normally
        _Logger.warning(
            "Max fast-forward iterations (%d) reached for %s, stopping.",
            max_iterations,
            entity_id,
        )
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
