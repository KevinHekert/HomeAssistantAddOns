import os
import json
import logging
from datetime import datetime, timedelta, timezone
from urllib import request, error

from db.samples import sample_exists, log_sample

_Logger = logging.getLogger(__name__)

SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN")


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
    """
    if not SUPERVISOR_TOKEN:
        _Logger.warning(
            "Geen SUPERVISOR_TOKEN gevonden; history sync voor %s wordt overgeslagen.",
            entity_id,
        )
        return

    now_utc = datetime.now(timezone.utc)

    if since is None:
        # Nog geen samples → bijvoorbeeld laatste 7 dagen binnenhalen
        start = now_utc - timedelta(days=7)
        _Logger.info(
            "Geen bestaande samples voor %s, history sync vanaf %s",
            entity_id,
            start,
        )
    else:
        # Klein beetje terug in de tijd om randgevallen mee te pakken
        start = since - timedelta(minutes=5)
        _Logger.info(
            "History sync voor %s vanaf %s (laatste sample was %s)",
            entity_id,
            start,
            since,
        )

    start_iso = start.astimezone(timezone.utc).isoformat()

    url = (
        f"http://supervisor/core/api/history/period/{start_iso}"
        f"?filter_entity_id={entity_id}"
    )

    req = request.Request(url)
    req.add_header("Authorization", f"Bearer {SUPERVISOR_TOKEN}")
    req.add_header("Content-Type", "application/json")

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
        return

    # HA-history structuur: lijst van lijsten; eerste lijst bevat states voor deze entity
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
            continue  # niet-numerieke states slaan we over

        unit = attributes.get("unit_of_measurement")

        # Dubbel-check op entity + timestamp
        if sample_exists(entity_id, ts):
            skipped += 1
            continue

        log_sample(entity_id, ts, value, unit)
        inserted += 1

    _Logger.info(
        "History sync voor %s afgerond: %d nieuwe, %d overgeslagen (bestonden al).",
        entity_id,
        inserted,
        skipped,
    )
