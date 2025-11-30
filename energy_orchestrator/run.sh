#!/usr/bin/with-contenv bashio

bashio::log.info "Starting Django Energy Orchestrator"

cd /app

# Eventueel migrations draaien (nu nog weinig effect)
python3 manage.py migrate --noinput || bashio::log.warning "Migrations failed (ok for first dev iterations)"

# Start Django dev server – prima voor interne add-on
python3 manage.py runserver 0.0.0.0:8099
