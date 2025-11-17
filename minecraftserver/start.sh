#!/bin/bash
set -e

FLASK_PORT="${FLASK_PORT:-8080}"
cd /opt/flask
echo "🚀 Starting Flask webserver on port ${FLASK_PORT} (waitress)..."
waitress-serve --listen=0.0.0.0:${FLASK_PORT} app:app &

cd /opt/bds
echo "🎮 Starting Bedrock server..."
exec /opt/bedrock-entry.sh "$@"