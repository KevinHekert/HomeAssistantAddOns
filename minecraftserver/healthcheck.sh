#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-/data}"
CONFIG_FILE="${DATA_DIR}/config/bedrock_for_ha_config.json"
STOP_MARKER="${DATA_DIR}/run/bedrock_server.stopped"

# EULA Standaard niet geaccepteerd (vereiste)
eula="false"
if [ -f "$CONFIG_FILE" ]; then
  eula="$(jq -r '.general.eula // false' "$CONFIG_FILE" 2>/dev/null || echo "false")"
fi

# Zolang de EULA NIET geaccepteerd is, beschouwen we de add-on als "gezond":
# UI werkt, gebruiker kan de EULA alsnog aanvinken.
case "${eula,,}" in
  true|1|yes|on)
    if [ -f "${STOP_MARKER}" ]; then
      exit 0
    fi
    # EULA geaccepteerd -> Bedrock hoort te draaien; healthcheck moet dat afdwingen
    echo ${eula}
    ;;
  *)
    # EULA niet geaccepteerd -> UI-only modus is OK voor Supervisor
    exit 0
    ;;
esac

# Als Eula geaccepteerd is, controleren of de server draait
transport="nethernet"
if [ -f "$CONFIG_FILE" ]; then
  transport="$(jq -r '.general.transport // "nethernet"' "$CONFIG_FILE" 2>/dev/null || echo "nethernet")"
fi

port="${SERVER_PORT:-19132}"

case "${transport,,}" in
  nethernet)
    # mc-monitor only speaks RakNet. Under NetherNet the same port carries the
    # TCP signalling socket instead, so probe that.
    timeout 3s bash -c "exec 3<>/dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1 || exit 1
    ;;
  *)
    timeout 3s /usr/local/bin/mc-monitor status-bedrock \
      --host 127.0.0.1 \
      --port "${port}" >/dev/null 2>&1 || exit 1
    ;;
esac
