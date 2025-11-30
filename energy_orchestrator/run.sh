#!/usr/bin/with-contenv bashio

bashio::log.info "Starting Energy Orchestrator (Flask)"

cd /app
exec python app.py