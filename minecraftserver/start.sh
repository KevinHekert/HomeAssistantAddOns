#!/bin/bash
set -e

FLASK_PORT="${FLASK_PORT:-8789}"
DATA_DIR="${DATA_DIR:-/data}"
RUNTIME_DIR="${DATA_DIR}/run"
BEDROCK_PID_FILE="${RUNTIME_DIR}/bedrock_server.pid"
STOP_MARKER="${RUNTIME_DIR}/bedrock_server.stopped"

BEDROCK_PID=""
SHUTDOWN_GRACE_SECONDS="${SHUTDOWN_GRACE_SECONDS:-30}"

start_bedrock_server() {
  echo "🎮 Starting Bedrock server..."
  mkdir -p "${RUNTIME_DIR}"
  rm -f "${STOP_MARKER}"
  /opt/bedrock-entry.sh "$@" &
  BEDROCK_PID=$!
  echo "${BEDROCK_PID}" >"${BEDROCK_PID_FILE}"
  echo "🧭 Bedrock PID saved to ${BEDROCK_PID_FILE}"
}

# Without this, Docker signals only this script. Bedrock keeps running until the
# stop grace period expires, so the next container start finds its port taken and
# bedrock_server fails to bind without logging anything.
shutdown_bedrock_server() {
  trap - SIGTERM SIGINT

  if [[ -n "${BEDROCK_PID}" ]] && kill -0 "${BEDROCK_PID}" 2>/dev/null; then
    echo "🛑 Stopping Bedrock server (pid ${BEDROCK_PID})..."
    kill -TERM "${BEDROCK_PID}" 2>/dev/null || true

    local deadline=$((SECONDS + SHUTDOWN_GRACE_SECONDS))
    while kill -0 "${BEDROCK_PID}" 2>/dev/null && ((SECONDS < deadline)); do
      sleep 0.5
    done

    if kill -0 "${BEDROCK_PID}" 2>/dev/null; then
      echo "⚠️ Bedrock did not exit in ${SHUTDOWN_GRACE_SECONDS}s; sending SIGKILL"
      kill -KILL "${BEDROCK_PID}" 2>/dev/null || true
      wait "${BEDROCK_PID}" 2>/dev/null || true
    fi

    echo "✅ Bedrock server stopped; port released"
  fi

  rm -f "${BEDROCK_PID_FILE}"
  exit 0
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

trap shutdown_bedrock_server SIGTERM SIGINT

cd /opt/flask
echo "🚀 Starting Flask webserver on port ${FLASK_PORT}..."
waitress-serve --listen=0.0.0.0:${FLASK_PORT} app:app &

# Start Bedrock server in the background so it can be controlled from the UI
cd /opt/bds
start_bedrock_server "$@"

# Keep the container alive and ensure the PID file reflects reality.
# Sleep in the background and wait on it so traps run immediately.
while true; do
  cleanup_stale_pid
  sleep 5 &
  wait $! || true
done
