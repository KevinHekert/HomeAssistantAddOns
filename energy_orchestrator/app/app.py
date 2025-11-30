import os
import json
import logging
import time
import threading

from urllib import request, error
from flask import Flask, render_template
#from sqlalchemy import create_engine, text, DateTime, Float, Integer, String
from sqlalchemy.exc import SQLAlchemyError
#from sqlalchemy.orm import declarative_base, Mapped, mapped_column, Session
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from datetime import datetime, timezone, timedelta
from db import Base, Sample



app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
_Logger = logging.getLogger(__name__)
_wind_logger_started = False


#DB Settings
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "username")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_NAME = os.environ.get("DB_NAME", "energy_orchestrator")

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(DB_URL, future=True)



#Entities
WIND_ENTITY_ID = os.environ.get("WIND_ENTITY_ID", "sensor.knmi_windsnelheid")


#Functies
def parse_ha_timestamp(value: str) -> datetime | None:
    """Parseer een ISO timestamp uit Home Assistant (met eventuele 'Z')."""
    if not value:
        return None
    # Home Assistant geeft meestal bijv. "2025-11-30T17:26:32.123456+00:00" of met 'Z'
    value = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        _Logger.error("Kan HA timestamp niet parsen: %s", value)
        return None
    
def test_db_connection():
    """Heel simpele check of MariaDB bereikbaar is."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _Logger.info("Verbinding met MariaDB geslaagd.")
    except SQLAlchemyError as e:
        _Logger.error("Fout bij verbinden met MariaDB: %s", e)

def init_db_schema():
    """Maak de tabellen aan als ze nog niet bestaan."""
    try:
        Base.metadata.create_all(engine)
        _Logger.info("Database schema bijgewerkt (samples).")
    except SQLAlchemyError as e:
        _Logger.error("Fout bij aanmaken schema in MariaDB: %s", e)


def get_wind_speed_from_ha():
    """Lees de actuele windsnelheid + eenheid uit Home Assistant via de Supervisor API."""
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        _Logger.warning("Geen SUPERVISOR_TOKEN gevonden in omgevingsvariabelen.")
        return None, None
    
    _Logger.info("Lezen windsnelheid van Home Assistant entiteit: %s", WIND_ENTITY_ID)
    url = f"http://supervisor/core/api/states/{WIND_ENTITY_ID}"

    req = request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    try:
        _Logger.debug("Verzoek sturen naar Home Assistant API: %s", url)
        with request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.URLError as e:
        _Logger.error("Fout bij verbinden met Home Assistant API: %s", e)
        return None, None
    except Exception:
        _Logger.error("Onverwachte fout bij ophalen van gegevens van Home Assistant.", exc_info=True)
        return None, None

    state = data.get("state")
    attributes = data.get("attributes", {})
    unit = attributes.get("unit_of_measurement")

    try:
        value = float(state)
    except (TypeError, ValueError):
        value = None

    return value, unit

def wind_logging_worker():
    """Achtergrondthread die periodiek samples sync't uit HA-history."""
    _Logger.info("Wind logging worker gestart.")
    while True:
        try:
            test_db_connection()
            init_db_schema()

            latest_ts = get_latest_sample_timestamp(WIND_ENTITY_ID)
            _Logger.info(
                "Huidige laatste timestamp voor %s vóór sync: %s",
                WIND_ENTITY_ID,
                latest_ts,
            )

            sync_history_for_entity(WIND_ENTITY_ID, latest_ts)
        except Exception as e:
            _Logger.error("Onverwachte fout in wind logging worker: %s", e)
        # Elke 5 minuten opnieuw syncen
        time.sleep(300)



def start_wind_logging_worker():
    global _wind_logger_started
    if _wind_logger_started:
        return
    thread = threading.Thread(target=wind_logging_worker, daemon=True)
    thread.start()
    _wind_logger_started = True
    _Logger.info("Wind logging worker thread gestart.")

def log_sample(entity_id: str, timestamp: datetime, value: float | None, unit: str | None) -> None:
    if value is None:
        _Logger.info("Geen waarde om op te slaan voor %s, overslaan.", entity_id)
        return

    try:
        with Session(engine) as session:
            sample = Sample(
                entity_id=entity_id,
                timestamp=timestamp,
                value=float(value),
                unit=unit,
            )
            session.add(sample)
            session.commit()
        _Logger.info(
            "Sample opgeslagen: entity=%s, ts=%s, value=%s, unit=%s",
            entity_id,
            timestamp,
            value,
            unit,
        )
    except SQLAlchemyError as e:
        _Logger.error("Fout bij opslaan van Sample voor %s: %s", entity_id, e)

def get_latest_sample_timestamp(entity_id: str) -> datetime | None:
    """Geef de laatste timestamp terug voor deze entiteit, of None als er geen samples zijn."""
    try:
        with Session(engine) as session:
            result = (
                session.query(Sample.timestamp)
                .filter(Sample.entity_id == entity_id)
                .order_by(Sample.timestamp.desc())
                .limit(1)
                .one_or_none()
            )
        if result is None:
            _Logger.info("Nog geen samples gevonden voor %s.", entity_id)
            return None

        latest_ts = result[0]
        _Logger.info("Laatste sample voor %s: %s", entity_id, latest_ts)
        return latest_ts
    except SQLAlchemyError as e:
        _Logger.error("Fout bij ophalen laatste sample voor %s: %s", entity_id, e)
        return None

def sample_exists(entity_id: str, timestamp: datetime) -> bool:
    """Check of er al een sample is voor deze entity + timestamp."""
    try:
        with Session(engine) as session:
            exists = (
                session.query(Sample.id)
                .filter(
                    Sample.entity_id == entity_id,
                    Sample.timestamp == timestamp,
                )
                .first()
                is not None
            )
        return exists
    except SQLAlchemyError as e:
        _Logger.error(
            "Fout bij controleren of sample bestaat voor %s @ %s: %s",
            entity_id,
            timestamp,
            e,
        )
        return True  # bij twijfel: liever niet dubbel inserten
    
def sync_history_for_entity(entity_id: str, since: datetime | None) -> None:
    """
    Sync alle history uit Home Assistant voor deze entity vanaf 'since' tot nu.

    - Haalt alle tussenliggende waardes op.
    - Voegt alleen nieuwe samples toe (op basis van entity_id + timestamp).
    """
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        _Logger.warning(
            "Geen SUPERVISOR_TOKEN gevonden; history sync voor %s wordt overgeslagen.",
            entity_id,
        )
        return

    # Startpunt bepalen
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
    req.add_header("Authorization", f"Bearer {token}")
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

@app.get("/")
def index():
    # Alleen actuele waarde ophalen voor de UI
    wind_speed, wind_unit = get_wind_speed_from_ha()

    return render_template(
        "index.html",
        wind_speed=wind_speed,
        wind_unit=wind_unit,
    )
    

if __name__ == "__main__":
    start_wind_logging_worker()
    app.run(host="0.0.0.0", port=8099, debug=False)
