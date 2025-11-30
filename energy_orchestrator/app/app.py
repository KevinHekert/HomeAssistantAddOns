import os
import json
import logging

from urllib import request, error

from flask import Flask, render_template

app = Flask(__name__)

_Logger = logging.getLogger(__name__)
# Voor nu hardcoded; later maken we dit configureerbaar
WIND_ENTITY_ID = os.environ.get("WIND_ENTITY_ID", "ssensor.knmi_windsnelheid")


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
        with request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.URLError:
        return None
    except Exception:
        return None

    # Probeer eerst state als float
    state = data.get("state")
    try:
        return float(state)
    except (TypeError, ValueError):
        return state  # eventueel string teruggeven

@app.get("/")
def index():
    wind_speed = get_wind_speed_from_ha()
    return render_template("index.html", wind_speed=wind_speed)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8099, debug=True)
