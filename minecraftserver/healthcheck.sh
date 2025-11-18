#!/bin/bash
set -e

CONFIG_FILE="/data/config/bedrock_for_ha_config.json"

# Default: nog geen config of geen eula veld -> beschouw als niet geaccepteerd
eula="false"
if [ -f "$CONFIG_FILE" ]; then
  eula="$(jq -r '.general.eula // false' "$CONFIG_FILE" 2>/dev/null || echo "false")"
fi

# Zolang de EULA NIET geaccepteerd is, beschouwen we de add-on als "gezond":
# UI werkt, gebruiker kan de EULA alsnog aanvinken.
case "${eula,,}" in
  true|1|yes|on)
    # EULA geaccepteerd -> Bedrock hoort te draaien; healthcheck moet dat afdwingen
    ;;
  *)
    # EULA niet geaccepteerd -> UI-only modus is OK voor Supervisor
    exit 0
    ;;
esac

# Vanaf hier: EULA = true, nu moet Bedrock bereikbaar zijn
timeout 3s /usr/local/bin/mc-monitor status-bedrock \
  --host 127.0.0.1 \
  --port "${SERVER_PORT:-19132}" >/dev/null 2>&1 || exit 1
