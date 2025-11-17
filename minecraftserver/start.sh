#!/bin/bash
set -e

FLASK_PORT="${FLASK_PORT:-8080}"

echo "🚀 Starting Flask webserver on port ${FLASK_PORT}..."
python3 /opt/flask/app.py &

echo "🎮 Starting Bedrock server..."
exec /opt/bedrock-entry.sh "$@"