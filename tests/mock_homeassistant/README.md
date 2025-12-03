# Mock Home Assistant API

This directory contains a lightweight Flask application that mocks the Home Assistant API for integration testing purposes.

## Purpose

The mock API simulates the following Home Assistant endpoints:
- `/api/` - API root (health check)
- `/api/states` - Get all entity states
- `/api/states/<entity_id>` - Get specific entity state
- `/api/history/period/<start_time>` - Get historical sensor data
- `/api/services/<domain>/<service>` - Call services

## Mock Sensors

The following sensors are mocked with realistic values:
- Wind speed
- Outdoor/indoor temperatures
- Flow/return temperatures
- Humidity, pressure
- Heat pump energy consumption
- DHW temperature and status

## Usage

This is automatically started by `docker-compose.test.yml` when running integration tests.

## Standalone Testing

You can run the mock API standalone:

```bash
cd tests/mock_homeassistant
python app.py
```

Then access it at http://localhost:8123/api/
