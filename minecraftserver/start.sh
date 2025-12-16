#!/bin/bash
set -e

FLASK_PORT="${FLASK_PORT:-8789}"
BEDROCK_PID_FILE="/run/bedrock_server.pid"

start_bedrock_server() {
  echo "🎮 Starting Bedrock server..."
  rm -f /run/bedrock_server.stopped
  /opt/bedrock-entry.sh "$@" &
  local bedrock_pid=$!
  echo "${bedrock_pid}" >"${BEDROCK_PID_FILE}"
  echo "🧭 Bedrock PID saved to ${BEDROCK_PID_FILE}"
}

cleanup_stale_pid() {
  if [[ -f "${BEDROCK_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${BEDROCK_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" && ! -d "/proc/${pid}" ]]; then
      rm -f "${BEDROCK_PID_FILE}"
    fi
  fi
}

cd /opt/flask
echo "🚀 Starting Flask webserver on port ${FLASK_PORT}..."
waitress-serve --listen=0.0.0.0:${FLASK_PORT} app:app &

# Start Bedrock server in the background so it can be controlled from the UI
cd /opt/bds
start_bedrock_server "$@"

# Keep the container alive and ensure the PID file reflects reality
while true; do
  cleanup_stale_pid
  sleep 5
done