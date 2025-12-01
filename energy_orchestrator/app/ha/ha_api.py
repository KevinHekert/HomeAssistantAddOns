import os
import json
import logging
from datetime import datetime, timedelta, timezone
from urllib import request, error
from urllib.parse import urlencode, quote

from db.samples import sample_exists, log_sample
from db.sync_state import update_sync_attempt

_Logger = logging.getLogger(__name__)

SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN")
BACKFILL_IF_NO_SAMPLES_DAYS = 14
MAX_WINDOW_DAYS = 1


def parse_state_to_float(state: str | None) -> float | None:
    """
    Parse a Home Assistant state value to a float.

    Handles:
    - Numeric strings (e.g., "23.5") -> float value
    - "on" -> 1.0
    - "off" -> 0.0
    - "true" -> 1.0
    - "false" -> 0.0
    - None or invalid values -> None
    """
    if state is None:
        return None

    state_lower = state.lower()

    if state_lower in ("on", "true"):
        return 1.0
    if state_lower in ("off", "false"):
        return 0.0

    try:
        return float(state)
    except (TypeError, ValueError):
        return None



def parse_ha_timestamp(value: str) -> datetime | None:
    """Parseer een ISO timestamp uit Home Assistant (met eventuele 'Z')."""
    if not value:
        return None
    value = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        _Logger.error("Kan HA timestamp niet parsen: %s", value)
        return None


def get_entity_state(entity_id: str) -> tuple[float | None, str | None]:
    """Lees actuele state + eenheid voor een entity vanuit HA."""
    if not SUPERVISOR_TOKEN:
        _Logger.warning("Geen SUPERVISOR_TOKEN gevonden; kan entity state niet lezen.")
        return None, None

    url = f"http://supervisor/core/api/states/{entity_id}"

    req = request.Request(url)
    req.add_header("Authorization", f"Bearer {SUPERVISOR_TOKEN}")
    req.add_header("Content-Type", "application/json")

    try:
        _Logger.debug("Verzoek sturen naar HA API (state) voor %s: %s", entity_id, url)
        with request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.URLError as e:
        _Logger.error("Fout bij state-opvraag voor %s: %s", entity_id, e)
        return None, None
    except Exception:
        _Logger.error(
            "Onverwachte fout bij state-opvraag voor %s.", entity_id, exc_info=True
        )
        return None, None

    state = data.get("state")
    attributes = data.get("attributes", {})
    unit = attributes.get("unit_of_measurement")

    value = parse_state_to_float(state)

    return value, unit


def sync_history_for_entity(entity_id: str, since: datetime | None) -> int:
    """
    Sync alle history uit Home Assistant voor deze entity.

    - Als er nog geen samples zijn (since is None):
      start = nu - BACKFILL_IF_NO_SAMPLES_DAYS
      end   = start + MAX_WINDOW_DAYS

    - Als er al samples zijn:
      start = since - 5 minuten
      end   = start + MAX_WINDOW_DAYS (maar niet voorbij nu)

    - Voegt alleen nieuwe samples toe (op basis van entity_id + timestamp).
    - Slaat altijd een sync-poging op in SyncStatus (ook bij geen data / fout).

    Returns:
        Number of newly inserted samples (0 if none found or error).
    """
    if not SUPERVISOR_TOKEN:
        _Logger.warning(
            "Geen SUPERVISOR_TOKEN gevonden; history sync voor %s wordt overgeslagen.",
            entity_id,
        )
        return 0

    now_utc = datetime.now(timezone.utc)

    # Normaliseer 'since' naar timezone-aware UTC (DB geeft vaak naive datetimes)
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    window = timedelta(days=MAX_WINDOW_DAYS)

    if since is None:
        # DB is leeg → begin 14 dagen terug en haal maar 1 dag op
        start = now_utc - timedelta(days=BACKFILL_IF_NO_SAMPLES_DAYS)
    else:
        # Wel samples → vanaf laatste sample, klein stukje terug voor veiligheid
        start = since - timedelta(minutes=5)

    # Eindtijd is start + window, maar nooit later dan nu
    end = start + window
    if end > now_utc:
        end = now_utc

    _Logger.debug(
        "History sync voor %s: start=%s, end=%s, now=%s",
        entity_id,
        start,
        end,
        now_utc,
    )

    start_iso = start.astimezone(timezone.utc).isoformat()
    end_iso = end.astimezone(timezone.utc).isoformat()

    start_encoded = quote(start_iso)

    query = urlencode(
        {
            "end_time": end_iso,
            "filter_entity_id": entity_id,
        }
    )

    url = f"http://supervisor/core/api/history/period/{start_encoded}?{query}"

    req = request.Request(url)
    req.add_header("Authorization", f"Bearer {SUPERVISOR_TOKEN}")
    req.add_header("Content-Type", "application/json")

    # Standaard: poging geregistreerd, nog niet succesvol
    update_sync_attempt(entity_id, end, success=False)

    try:
        _Logger.debug("History-verzoek naar Home Assistant API: %s", url)
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.URLError as e:
        _Logger.error("Fout bij history-opvraag voor %s: %s", entity_id, e)
        return 0
    except Exception:
        _Logger.error(
            "Onverwachte fout bij history-opvraag voor %s.", entity_id, exc_info=True
        )
        return 0

    if not data:
        _Logger.debug("Geen history-data ontvangen voor %s.", entity_id)
        # poging blijft wel geregistreerd, maar geen success-flag
        return 0

    states = data[0] if isinstance(data[0], list) else data
    _Logger.debug("Aantal historypunten voor %s ontvangen: %d", entity_id, len(states))

    inserted = 0
    skipped = 0

    for state_obj in states:
        raw_state = state_obj.get("state")
        attributes = state_obj.get("attributes", {})

        ts_str = state_obj.get("last_updated") or state_obj.get("last_changed")
        ts = parse_ha_timestamp(ts_str)
        if ts is None:
            continue

        value = parse_state_to_float(raw_state)
        if value is None:
            continue

        unit = attributes.get("unit_of_measurement")

        if sample_exists(entity_id, ts):
            skipped += 1
            continue

        log_sample(entity_id, ts, value, unit)
        inserted += 1

    # Als we hier zijn, was de call inhoudelijk oké; we markeren deze poging als succesvol
    update_sync_attempt(entity_id, end, success=True)

    # Summary log: only log at INFO level if new samples were inserted
    if inserted > 0:
        _Logger.info(
            "Opgehaald en opgeslagen: %d metingen van %s.",
            inserted,
            entity_id,
        )
    else:
        _Logger.debug(
            "History sync voor %s afgerond: %d nieuwe, %d overgeslagen (bestonden al).",
            entity_id,
            inserted,
            skipped,
        )

    return inserted


