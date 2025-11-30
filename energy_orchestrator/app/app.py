import os
import json
import logging

from urllib import request, error
from flask import Flask, render_template
from sqlalchemy import create_engine, text, DateTime, Float, Integer
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from datetime import datetime

app = Flask(__name__)
_Logger = logging.getLogger(__name__)


#DB Settings
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "username")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_NAME = os.environ.get("DB_NAME", "energy_orchestrator")

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(DB_URL, future=True)

Base = declarative_base()

class WindSample(Base):
    __tablename__ = "wind_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    value: Mapped[float] = mapped_column(Float)


#Entities
WIND_ENTITY_ID = os.environ.get("WIND_ENTITY_ID", "sensor.knmi_windsnelheid")


#Functies
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
        _Logger.info("Database schema bijgewerkt (wind_samples).")
    except SQLAlchemyError as e:
        _Logger.error("Fout bij aanmaken schema in MariaDB: %s", e)


def get_wind_speed_from_ha():
    """Lees de actuele windsnelheid uit Home Assistant via de Supervisor API."""
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        _Logger.warning("Geen SUPERVISOR_TOKEN gevonden in omgevingsvariabelen.")
        return None
    
    _Logger.info("Lezen windsnelheid van Home Assistant entiteit: %s", WIND_ENTITY_ID)
    url = f"http://supervisor/core/api/states/{WIND_ENTITY_ID}"

    req = request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    try:
        _Logger.debug("Verzoek sturen naar Home Assistant API.")
        with request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.URLError:
        _Logger.error("Fout bij verbinden met Home Assistant API.")
        return None
    except Exception:
        _Logger.error("Onverwachte fout bij ophalen van gegevens van Home Assistant.", exc_info=True)
        return None

    # Probeer eerst state als float
    state = data.get("state")
    try:
        return float(state)
    except (TypeError, ValueError):
        return state  # eventueel string teruggeven

@app.get("/")
def index():
    test_db_connection()
    init_db_schema()
    wind_speed = get_wind_speed_from_ha()
    return render_template("index.html", wind_speed=wind_speed)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8099, debug=True)
