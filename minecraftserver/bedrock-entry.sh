#!/bin/bash
set -eo pipefail

# =========================
#  Bedrock STARTER (no download)
#  - Reads options from /data/options.json
#  - Applies server.properties via set-property
#  - Builds permissions/allowlist (from options + env fallbacks)
#  - Starts pre-bundled binary at /opt/bds/bedrock_server-${VERSION}
# =========================

#Check symlinks

LINKS=(
  "/opt/bds/worlds:/data/worlds"
  "/opt/bds/server.properties:/data/server.properties"
  "/opt/bds/allowlist.json:/data/allowlist.json"
  "/opt/bds/whitelist.json:/data/whitelist.json"
  "/opt/bds/permissions.json:/data/permissions.json"
)

echo "🔗 Checking Bedrock symlinks..."

for entry in "${LINKS[@]}"; do
  target="${entry%%:*}"     # Left of :
  source="${entry##*:}"     # Right of :

  if [ -L "$target" ]; then
    echo "✔️ Symlink exists: $target → $(readlink "$target")"
  else
    echo "➕ Creating symlink: $target → $source"
    ln -s "$source" "$target"
  fi
done

echo "✨ Symlink check complete."

# --- Ensure /data/worlds exists ---
if [ ! -d /data/worlds ]; then
  echo "📁 Creating /data/worlds..."
  mkdir -p /data/worlds
  chmod 0777 /data/worlds
fi


# ---------- helpers ----------
isTrue() { case "${1,,}" in true|on|1|yes) return 0 ;; *) return 1 ;; esac; }
lower_bool() { case "${1,,}" in true|1|on|yes) echo "true" ;; *) echo "false" ;; esac; }

# JSON helpers
OPT_FILE="/data/config/bedrock_for_ha_config.json"
optn() { jq -r "$1 // empty" "$OPT_FILE" 2>/dev/null; }                       # nested path, e.g. '.world.gamemode'
optf() { jq -r --arg k "$1" '.[$k] // empty' "$OPT_FILE" 2>/dev/null; }       # flat key, e.g. 'gamemode'
first_nonempty() { for v in "$@"; do [[ -n "$v" ]] && { echo "$v"; return; }; done; echo ""; }

jq_safe_array_file() {
  local f="$1"
  if [[ ! -f "$f" ]] || ! jq -e . "$f" >/dev/null 2>&1; then
    echo "[]" > "$f"
  elif ! jq -e 'type=="array"' "$f" >/dev/null 2>&1; then
    jq -c '[.]' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  fi
}

# ---------- debug ----------
if [[ ${DEBUG^^} = TRUE ]]; then
  set -x
  echo "DEBUG: running as $(id -a) with $(ls -ld /data)"
  echo "       cwd=$(pwd)"
fi

# ---------- EULA ----------
if [[ ${EULA^^} != TRUE ]]; then
  echo
  echo "EULA must be set to TRUE to indicate agreement with the Minecraft End User License"
  echo "See https://minecraft.net/terms"
  echo "Current value is '${EULA}'"
  echo
  exit 1
fi

# ---------- Determine VERSION & binary path (pre-baked at build time) ----------
: "${VERSION:=$(cat /etc/bds-version 2>/dev/null || true)}"
BIN_DIR="/opt/bds"
BIN_PATH="${BIN_DIR}/bedrock_server-${VERSION}"

if [[ -z "$VERSION" ]]; then
  echo "ERROR: VERSION is empty and /etc/bds-version not found. This image expects a pre-bundled Bedrock binary."
  exit 2
fi
if [[ ! -x "$BIN_PATH" ]]; then
  echo "ERROR: Binary not found: $BIN_PATH"
  echo "       Ensure the image baked the server as /opt/bds/bedrock_server-${VERSION}"
  exit 2
fi

# ---------- allow/white list ----------
allowListUsers=${ALLOW_LIST_USERS:-${WHITE_LIST_USERS}}
if [ -n "$allowListUsers" ]; then
  echo "Setting allow list"
  for f in whitelist.json allowlist.json; do
    [ -f "$f" ] && rm -rf "$f"
    jq -n --arg users "$allowListUsers" '$users | split(",") | map({"name": .})' > "$f"
  done
  export WHITE_LIST=true
  export ALLOW_LIST=true
fi

# ---------- options → ENV (nested with flat fallbacks) ----------
# GENERAL
export SERVER_NAME="${SERVER_NAME:-$(first_nonempty "$(optn '.general.server_name')" "$(optf 'server_name')")}"
export SERVER_PORT="${SERVER_PORT:-$(first_nonempty "$(optn '.general.server_port')" "$(optf 'server_port')")}"
export SERVER_PORT_V6="${SERVER_PORT_V6:-$(first_nonempty "$(optn '.general.server_port_v6')" "$(optf 'server_port_v6')")}"
export ONLINE_MODE="$(lower_bool "${ONLINE_MODE:-$(first_nonempty "$(optn '.general.online_mode')" "$(optf 'online_mode')")}")"
export EMIT_SERVER_TELEMETRY="$(lower_bool "${EMIT_SERVER_TELEMETRY:-$(first_nonempty "$(optn '.general.emit_server_telemetry')" "$(optf 'emit_server_telemetry')")}")"
export ENABLE_LAN_VISIBILITY="$(lower_bool "${ENABLE_LAN_VISIBILITY:-$(first_nonempty "$(optn '.general.enable_lan_visibility')" "$(optf 'enable_lan_visibility')")}")"

# WORLD
export LEVEL_NAME="${LEVEL_NAME:-$(first_nonempty "$(optn '.world.level_name')" "$(optf 'level_name')")}"
export LEVEL_SEED="${LEVEL_SEED:-$(first_nonempty "$(optn '.world.level_seed')" "$(optf 'level_seed')")}"
export LEVEL_TYPE="${LEVEL_TYPE:-$(first_nonempty "$(optn '.world.level_type')" "$(optf 'level_type')")}"
export GAMEMODE="${GAMEMODE:-$(first_nonempty "$(optn '.world.gamemode')" "$(optf 'gamemode')")}"
export DIFFICULTY="${DIFFICULTY:-$(first_nonempty "$(optn '.world.difficulty')" "$(optf 'difficulty')")}"
export ALLOW_CHEATS="$(lower_bool "${ALLOW_CHEATS:-$(first_nonempty "$(optn '.world.allow_cheats')" "$(optf 'allow_cheats')")}")"

# PLAYERS
export MAX_PLAYERS="${MAX_PLAYERS:-$(first_nonempty "$(optn '.players.max_players')" "$(optf 'max_players')")}"
export WHITE_LIST="$(lower_bool "${WHITE_LIST:-$(first_nonempty "$(optn '.players.white_list')" "$(optf 'white_list')")}")"
export ALLOW_LIST="$(lower_bool "${ALLOW_LIST:-$(first_nonempty "$(optn '.players.allow_list')" "$(optf 'allow_list')")}")"
export DEFAULT_PLAYER_PERMISSION_LEVEL="${DEFAULT_PLAYER_PERMISSION_LEVEL:-$(first_nonempty "$(optn '.players.default_player_permission_level')" "$(optf 'default_player_permission_level')")}"
export TEXTUREPACK_REQUIRED="$(lower_bool "${TEXTUREPACK_REQUIRED:-$(first_nonempty "$(optn '.players.texturepack_required')" "$(optf 'texturepack_required')")}")"

# PERFORMANCE
export VIEW_DISTANCE="${VIEW_DISTANCE:-$(first_nonempty "$(optn '.performance.view_distance')" "$(optf 'view_distance')")}"
export TICK_DISTANCE="${TICK_DISTANCE:-$(first_nonempty "$(optn '.performance.tick_distance')" "$(optf 'tick_distance')")}"
export PLAYER_IDLE_TIMEOUT="${PLAYER_IDLE_TIMEOUT:-$(first_nonempty "$(optn '.performance.player_idle_timeout')" "$(optf 'player_idle_timeout')")}"
export MAX_THREADS="${MAX_THREADS:-$(first_nonempty "$(optn '.performance.max_threads')" "$(optf 'max_threads')")}"
export COMPRESSION_THRESHOLD="${COMPRESSION_THRESHOLD:-$(first_nonempty "$(optn '.performance.compression_threshold')" "$(optf 'compression_threshold')")}"

# ANTI_CHEAT
export SERVER_AUTHORITATIVE_MOVEMENT="${SERVER_AUTHORITATIVE_MOVEMENT:-$(first_nonempty "$(optn '.anti_cheat.server_authoritative_movement')" "$(optf 'server_authoritative_movement')")}"
export SERVER_AUTHORITATIVE_BLOCK_BREAKING="$(lower_bool "${SERVER_AUTHORITATIVE_BLOCK_BREAKING:-$(first_nonempty "$(optn '.anti_cheat.server_authoritative_block_breaking')" "$(optf 'server_authoritative_block_breaking')")}")"
export PLAYER_MOVEMENT_SCORE_THRESHOLD="${PLAYER_MOVEMENT_SCORE_THRESHOLD:-$(first_nonempty "$(optn '.anti_cheat.player_movement_score_threshold')" "$(optf 'player_movement_score_threshold')")}"
export PLAYER_MOVEMENT_DISTANCE_THRESHOLD="${PLAYER_MOVEMENT_DISTANCE_THRESHOLD:-$(first_nonempty "$(optn '.anti_cheat.player_movement_distance_threshold')" "$(optf 'player_movement_distance_threshold')")}"
export PLAYER_MOVEMENT_DURATION_THRESHOLD_IN_MS="${PLAYER_MOVEMENT_DURATION_THRESHOLD_IN_MS:-$(first_nonempty "$(optn '.anti_cheat.player_movement_duration_threshold_in_ms')" "$(optf 'player_movement_duration_threshold_in_ms')")}"
export CORRECT_PLAYER_MOVEMENT="$(lower_bool "${CORRECT_PLAYER_MOVEMENT:-$(first_nonempty "$(optn '.anti_cheat.correct_player_movement')" "$(optf 'correct_player_movement')")}")"

# ---------- Build permissions.json from UI (role_assignments) + env fallbacks ----------
ensure_permissions_file() {
  if [[ ! -f permissions.json ]] || ! jq -e . permissions.json >/dev/null 2>&1; then
    echo "[]" > permissions.json
  fi
}

assignments_json="$(jq -c '.players.role_assignments // []' "$OPT_FILE" 2>/dev/null || echo '[]')"

env_to_items() {
  local csv="$1" role="$2"
  [[ -z "$csv" ]] && return 0
  awk -v RS=',' -v role="$role" '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); if(length($0)) printf("{\"xuid\":\"%s\",\"role\":\"%s\"}\n",$0,role)}' <<< "$csv"
}

tmp="$(mktemp)"
{
  jq -c '.[] | {"xuid": (.xuid|tostring), "role": (.role|tostring)}' <<< "$assignments_json"
  env_to_items "${OPS}"      "operator"
  env_to_items "${MEMBERS}"  "member"
  env_to_items "${VISITORS}" "visitor"
} | jq -s '
  map(.role |= ( . as $r |
    if ($r=="operator" or $r=="member" or $r=="visitor") then $r else "member" end)) |
  (reduce .[] as $i ({}; .[$i.xuid] = {xuid:$i.xuid, permission:$i.role})) |
  to_entries | map(.value)
' > "$tmp" && mv "$tmp" permissions.json
ensure_permissions_file
echo "✅ permissions.json generated"

# ---------- Apply server.properties from ENV via definitions ----------
PROP_FILE="/data/server.properties"
touch "$PROP_FILE"

if [ -f /etc/bds-property-definitions.json ]; then
  set-property --file "$PROP_FILE" --bulk /etc/bds-property-definitions.json
else
  echo "WARN: /etc/bds-property-definitions.json missing; skipping bulk apply"
fi

# ---------- Pre-start info ----------
echo "📜 server.properties (excerpt):"
echo "-------------------------------------------"
if [ -f "$PROP_FILE" ]; then
  grep -E '^(server-name|gamemode|difficulty|level-name|default-player-permission-level|view-distance|tick-distance|online-mode|server-port|max-players)' "$PROP_FILE" || echo "⚠️ Geen properties gevonden"
else
  echo "⚠️ $PROP_FILE bestaat nog niet!"
fi
echo "-------------------------------------------"

# ---------- Start ----------
export LD_LIBRARY_PATH="${BIN_DIR}"
echo Library path: ${LD_LIBRARY_PATH:-"(not set)"}

echo "🚀 Starting Bedrock ${VERSION}"
if [ -f /usr/local/bin/box64 ] ; then
  exec box64 "${BIN_PATH}"
else
  exec "${BIN_PATH}"
fi
