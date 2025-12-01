import logging
from flask import Flask, render_template, jsonify
from ha.ha_api import get_entity_state
from workers import start_sensor_logging_worker
from db.resample import resample_all_categories_to_5min
from db.core import init_db_schema
from db.sensor_config import sync_sensor_mappings



app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
_Logger = logging.getLogger(__name__)

# Voor nu één sensor; later lijst vanuit config/integratie
WIND_ENTITY_ID = "sensor.knmi_windsnelheid"


@app.get("/")
def index():
    # Alleen actuele waarde ophalen voor de UI
    wind_speed, wind_unit = get_entity_state(WIND_ENTITY_ID)

    return render_template(
        "index.html",
        wind_speed=wind_speed,
        wind_unit=wind_unit,
    )


@app.post("/resample")
def trigger_resample():
    """Trigger resampling of all categories to 5-minute slots."""
    try:
        _Logger.info("Resample triggered via UI")
        resample_all_categories_to_5min()
        return jsonify({"status": "success", "message": "Resampling completed successfully"})
    except Exception as e:
        _Logger.error("Error during resampling: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # Initialize database schema and sensor mappings before starting workers
    init_db_schema()
    sync_sensor_mappings()
    start_sensor_logging_worker()
    app.run(host="0.0.0.0", port=8099, debug=False)
