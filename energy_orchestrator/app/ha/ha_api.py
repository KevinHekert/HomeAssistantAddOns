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
MAX_WINDOW_DAYS = 1
BACKFILL_HORIZON_DAYS = 100

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

    try:
        value = float(state)
    except (TypeError, ValueError):
        value = None

    return value, unit


def sync_history_for_entity(entity_id: str, since: datetime | None) -> None:
    """
    Sync alle history uit Home Assistant voor deze entity vanaf 'since' tot nu.

    - Haalt alle tussenliggende waardes op.
    - Voegt alleen nieuwe samples toe (op basis van entity_id + timestamp).
    - Slaat altijd een sync-poging op in SyncStatus (ook bij geen data / fout).
    """
    if not SUPERVISOR_TOKEN:
        _Logger.warning(
            "Geen SUPERVISOR_TOKEN gevonden; history sync voor %s wordt overgeslagen.",
            entity_id,
        )
        return

    now_utc = datetime.now(timezone.utc)

    if since is None:
        # We willen "BACKFILL_HORIZON_DAYS" terug, maar beperken de window per request
        desired_start = now_utc - timedelta(days=BACKFILL_HORIZON_DAYS)
    else:
        # Voor incremental: vanaf laatste sample, klein stukje terug voor veiligheid
        desired_start = since - timedelta(minutes=5)

    # Nu clampen we de start zodat de span nooit groter wordt dan MAX_WINDOW_DAYS
    max_span = timedelta(days=MAX_WINDOW_DAYS)
    if now_utc - desired_start > max_span:
        start = now_utc - max_span
    else:
        start = desired_start

    _Logger.info(
        "History sync voor %s: desired_start=%s, effective_start=%s, now=%s",
        entity_id,
        desired_start,
        start,
        now_utc,
    )

    start_iso = start.astimezone(timezone.utc).isoformat()
    end_iso = now_utc.astimezone(timezone.utc).isoformat()

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
    update_sync_attempt(entity_id, now_utc, success=False)

    try:
        _Logger.info("History-verzoek naar Home Assistant API: %s", url)
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.URLError as e:
        _Logger.error("Fout bij history-opvraag voor %s: %s", entity_id, e)
        return
    except Exception:
        _Logger.error(
            "Onverwachte fout bij history-opvraag voor %s.", entity_id, exc_info=True
        )
        return

    if not data:
        _Logger.info("Geen history-data ontvangen voor %s.", entity_id)
        # poging blijft wel geregistreerd, maar geen success-flag
        return

    states = data[0] if isinstance(data[0], list) else data
    _Logger.info("Aantal historypunten voor %s ontvangen: %d", entity_id, len(states))

    inserted = 0
    skipped = 0

    for state_obj in states:
        raw_state = state_obj.get("state")
        attributes = state_obj.get("attributes", {})

        ts_str = state_obj.get("last_updated") or state_obj.get("last_changed")
        ts = parse_ha_timestamp(ts_str)
        if ts is None:
            continue

        try:
            value = float(raw_state)
        except (TypeError, ValueError):
            continue

        unit = attributes.get("unit_of_measurement")

        if sample_exists(entity_id, ts):
            skipped += 1
            continue

        log_sample(entity_id, ts, value, unit)
        inserted += 1

    # Als we hier zijn, was de call inhoudelijk oké; we markeren deze poging als succesvol
    update_sync_attempt(entity_id, now_utc, success=True)

    _Logger.info(
        "History sync voor %s afgerond: %d nieuwe, %d overgeslagen (bestonden al).",
        entity_id,
        inserted,
        skipped,
    )

