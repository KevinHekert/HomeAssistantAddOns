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

# Configurable sleep intervals (can be set via environment variables)
SENSOR_SYNC_INTERVAL = int(os.environ.get("SENSOR_SYNC_INTERVAL", "1"))
SENSOR_LOOP_INTERVAL = int(os.environ.get("SENSOR_LOOP_INTERVAL", "1"))

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


def _ensure_timezone_aware(dt: datetime | None) -> datetime | None:
    """Ensure a datetime is timezone-aware (UTC).
    
    Database often returns naive datetimes, so we normalize them to UTC.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _sync_entity(entity_id: str) -> None:
    """
    Voer sync-cyclus(sen) uit voor een enkele entity.

    When no historic data is found and we haven't caught up to yesterday yet,
    loop faster (without the normal 5-second delay between iterations) until
    data is found or we reach today-1.

    Also handles gaps in existing data: when samples exist but the last sample
    is more than 24 hours ago and no new data is found, fast-forward through
    the gap by checking subsequent 24-hour windows until data is found or
    we reach yesterday.
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

        # Determine effective_since: use the most recent of latest_ts or last_attempt.
        # When last_attempt is newer than latest_ts, it indicates a gap in the data
        # (we've already tried to sync beyond the last sample but found no new data).
        # Using last_attempt in this case allows us to progress through the gap.
        if latest_ts is not None:
            effective_since = latest_ts
            if status is not None and status.last_attempt is not None:
                latest_ts_aware = _ensure_timezone_aware(latest_ts)
                last_attempt_aware = _ensure_timezone_aware(status.last_attempt)
                
                if last_attempt_aware > latest_ts_aware:
                    effective_since = status.last_attempt
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

        # Check if we should continue fast-forwarding when no samples were inserted
        if inserted == 0:
            effective_since_aware = _ensure_timezone_aware(effective_since)
            if effective_since_aware is not None:
                # If effective_since is before yesterday, fast-forward without delay
                # This applies both when:
                # 1. No samples exist yet (latest_ts is None) - initial backfill
                # 2. Samples exist but there's a gap > 24h (latest_ts is not None)
                if effective_since_aware < yesterday:
                    if latest_ts is None:
                        _Logger.debug(
                            "No data found for %s (initial backfill) and not yet caught up to yesterday, fast-forwarding...",
                            entity_id,
                        )
                    else:
                        _Logger.debug(
                            "No data found for %s (gap in data, last sample: %s) and not yet caught up to yesterday, fast-forwarding...",
                            entity_id,
                            latest_ts,
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
        time.sleep(SENSOR_SYNC_INTERVAL)
        break
    else:
        # Max iterations reached - log warning and continue normally
        _Logger.warning(
            "Max fast-forward iterations (%d) reached for %s, stopping.",
            max_iterations,
            entity_id,
        )
        time.sleep(SENSOR_SYNC_INTERVAL)


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

        # Voor testing: minimaal wachten tussen loops
        time.sleep(1)


def start_sensor_logging_worker():
    global _sensor_worker_started
    if _sensor_worker_started:
        return

    thread = threading.Thread(target=sensor_logging_worker, daemon=True)
    thread.start()
    _sensor_worker_started = True
    _Logger.info("Sensor logging worker thread gestart.")
