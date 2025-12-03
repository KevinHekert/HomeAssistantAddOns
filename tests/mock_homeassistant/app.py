"""
Mock Home Assistant API Server for Integration Testing

This mock server simulates the Home Assistant API endpoints that the
Energy Orchestrator add-on depends on for sensor data retrieval.
"""

from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# Mock sensor states
SENSORS = {
    "sensor.knmi_windsnelheid": {"state": "3.5", "unit": "m/s", "friendly_name": "Wind Speed"},
    "sensor.smile_outdoor_temperature": {"state": "8.2", "unit": "°C", "friendly_name": "Outdoor Temperature"},
    "sensor.opentherm_water_temperature": {"state": "45.3", "unit": "°C", "friendly_name": "Flow Temperature"},
    "sensor.opentherm_return_temperature": {"state": "38.7", "unit": "°C", "friendly_name": "Return Temperature"},
    "sensor.knmi_luchtvochtigheid": {"state": "75", "unit": "%", "friendly_name": "Humidity"},
    "sensor.knmi_luchtdruk": {"state": "1013.25", "unit": "hPa", "friendly_name": "Air Pressure"},
    "sensor.extra_total": {"state": "125.8", "unit": "kWh", "friendly_name": "Heat Pump Total Energy"},
    "sensor.opentherm_dhw_temperature": {"state": "55.0", "unit": "°C", "friendly_name": "DHW Temperature"},
    "sensor.anna_temperature": {"state": "21.5", "unit": "°C", "friendly_name": "Indoor Temperature"},
    "sensor.anna_setpoint": {"state": "21.0", "unit": "°C", "friendly_name": "Target Temperature"},
    "binary_sensor.dhw_active": {"state": "off", "friendly_name": "DHW Active"},
}


@app.route('/api/', methods=['GET'])
def api_root():
    """Health check endpoint"""
    return jsonify({
        "message": "Mock Home Assistant API",
        "version": "2024.1.0"
    })


@app.route('/api/states', methods=['GET'])
def get_all_states():
    """Get all sensor states"""
    states = []
    for entity_id, data in SENSORS.items():
        states.append({
            "entity_id": entity_id,
            "state": data["state"],
            "attributes": {
                "unit_of_measurement": data.get("unit"),
                "friendly_name": data["friendly_name"]
            },
            "last_changed": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        })
    return jsonify(states)


@app.route('/api/states/<path:entity_id>', methods=['GET'])
def get_entity_state(entity_id):
    """Get a specific entity state"""
    if entity_id not in SENSORS:
        return jsonify({"error": "Entity not found"}), 404
    
    data = SENSORS[entity_id]
    return jsonify({
        "entity_id": entity_id,
        "state": data["state"],
        "attributes": {
            "unit_of_measurement": data.get("unit"),
            "friendly_name": data["friendly_name"]
        },
        "last_changed": datetime.utcnow().isoformat(),
        "last_updated": datetime.utcnow().isoformat()
    })


@app.route('/api/history/period/<start_time>', methods=['GET'])
def get_history(start_time):
    """Get historical sensor data"""
    entity_ids = request.args.get('filter_entity_id', '').split(',')
    end_time = request.args.get('end_time')
    
    # Generate mock historical data
    history = {}
    for entity_id in entity_ids:
        if entity_id and entity_id in SENSORS:
            sensor_data = SENSORS[entity_id]
            history[entity_id] = []
            
            # Generate 24 hours of data points (every 5 minutes)
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            for i in range(288):  # 24 * 60 / 5 = 288 data points
                timestamp = start + timedelta(minutes=i*5)
                
                # Add some variation to the base value
                base_value = float(sensor_data["state"])
                variation = random.uniform(-0.5, 0.5) * base_value * 0.1
                value = base_value + variation
                
                history[entity_id].append({
                    "state": str(round(value, 2)),
                    "last_changed": timestamp.isoformat(),
                    "attributes": {
                        "unit_of_measurement": sensor_data.get("unit"),
                        "friendly_name": sensor_data["friendly_name"]
                    }
                })
    
    return jsonify(history)


@app.route('/api/services/<domain>/<service>', methods=['POST'])
def call_service(domain, service):
    """Mock service calls"""
    return jsonify({
        "message": f"Service {domain}.{service} called",
        "success": True
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8123, debug=True)
