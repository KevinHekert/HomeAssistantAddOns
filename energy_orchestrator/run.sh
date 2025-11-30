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

cd /app
exec python app.py
