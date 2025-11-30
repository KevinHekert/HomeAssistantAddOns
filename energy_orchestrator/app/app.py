import logging
from flask import Flask, render_template
from ha.ha_api import get_entity_state
from workers.wind import start_wind_logging_worker

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


if __name__ == "__main__":
    start_wind_logging_worker()
    app.run(host="0.0.0.0", port=8099, debug=False)
