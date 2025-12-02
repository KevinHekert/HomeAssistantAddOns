#!/usr/bin/with-contenv bashio

bashio::log.info "Starting Energy Orchestrator (Flask)"

# DB-config uit add-on configuratie
DB_HOST="$(bashio::config 'db_host')"
DB_USER="$(bashio::config 'db_user')"
DB_PASSWORD="$(bashio::config 'db_password')"
DB_NAME="$(bashio::config 'db_name')"

bashio::log.info "DB host: ${DB_HOST}, user: ${DB_USER}, name: ${DB_NAME}"

export DB_HOST
export DB_USER
export DB_PASSWORD
export DB_NAME

# Sensor entity IDs uit add-on configuratie
export WIND_ENTITY_ID="$(bashio::config 'wind_entity_id')"
export OUTDOOR_TEMP_ENTITY_ID="$(bashio::config 'outdoor_temp_entity_id')"
export FLOW_TEMP_ENTITY_ID="$(bashio::config 'flow_temp_entity_id')"
export RETURN_TEMP_ENTITY_ID="$(bashio::config 'return_temp_entity_id')"
export HUMIDITY_ENTITY_ID="$(bashio::config 'humidity_entity_id')"
export PRESSURE_ENTITY_ID="$(bashio::config 'pressure_entity_id')"
export HP_KWH_TOTAL_ENTITY_ID="$(bashio::config 'hp_kwh_total_entity_id')"
export DHW_TEMP_ENTITY_ID="$(bashio::config 'dhw_temp_entity_id')"
export INDOOR_TEMP_ENTITY_ID="$(bashio::config 'indoor_temp_entity_id')"
export TARGET_TEMP_ENTITY_ID="$(bashio::config 'target_temp_entity_id')"
export DHW_ACTIVE_ENTITY_ID="$(bashio::config 'dhw_active_entity_id')"

# Sync configuration options
export BACKFILL_DAYS="$(bashio::config 'backfill_days')"
export SYNC_WINDOW_DAYS="$(bashio::config 'sync_window_days')"
export SENSOR_SYNC_INTERVAL="$(bashio::config 'sensor_sync_interval')"
export SENSOR_LOOP_INTERVAL="$(bashio::config 'sensor_loop_interval')"

bashio::log.info "Sync config: backfill_days=${BACKFILL_DAYS}, sync_window_days=${SYNC_WINDOW_DAYS}, sensor_sync_interval=${SENSOR_SYNC_INTERVAL}, sensor_loop_interval=${SENSOR_LOOP_INTERVAL}"

cd /app
exec python app.py
