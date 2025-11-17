import json
import os
from copy import deepcopy

from flask import Flask, request, redirect, url_for, render_template_string, flash, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
)
app.secret_key = "minecraftserver-for-ha"  # alleen voor flash-messages, mag random zijn

PERMISSIONS_FILE = "/opt/bds/permissions.json"
CONFIG_DIR = "/data/config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "bedrock_for_ha_config.json")
WORLDS_DIR = "/data/worlds"

# ---- Default config (zelfde structuur als 'options' in config.yaml) ----
DEFAULT_CONFIG = {
    "general": {
        "server_name": "HomeAssistantMinecraftServer",
        "server_port": 19132,
        "server_port_v6": 19133,
        "online_mode": True,
        "emit_server_telemetry": False,
        "enable_lan_visibility": True,
    },
    "world": {
        "level_name": "HomeAssistant",
        "level_seed": "-1251937210",
        "level_type": "DEFAULT",
        "gamemode": "survival",
        "difficulty": "normal",
        "allow_cheats": False,
    },
    "players": {
        "max_players": 20,
        "white_list": False,
        "allow_list": False,
        "default_player_permission_level": "visitor",
        "texturepack_required": False,
        "role_assignments": [],
    },
    "performance": {
        "view_distance": 32,
        "tick_distance": 12,
        "player_idle_timeout": 0,
        "max_threads": 0,
        "compression_threshold": 1,
    },
    "anti_cheat": {
        "server_authoritative_movement": "server-auth-with-rewind",
        "server_authoritative_block_breaking": True,
        "player_movement_score_threshold": 20,
        "player_movement_distance_threshold": 0.3,
        "player_movement_duration_threshold_in_ms": 500,
        "correct_player_movement": False,
    },
}

# ---- Helpers ----


def deep_merge(defaults, overrides):
    """Recursively merge two dicts: overrides op defaults."""
    result = deepcopy(defaults)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def ensure_dirs():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(WORLDS_DIR, exist_ok=True)


def ensure_config_file():
    ensure_dirs()
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, sort_keys=True)


def load_config():
    ensure_config_file()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    return deep_merge(DEFAULT_CONFIG, data)


def save_config(config):
    ensure_dirs()
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def list_worlds():
    ensure_dirs()
    worlds = []
    for name in os.listdir(WORLDS_DIR):
        full = os.path.join(WORLDS_DIR, name)
        if os.path.isdir(full) and not name.startswith("."):
            worlds.append(name)
    return sorted(worlds)


def to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in ("true", "1", "on", "yes")


def to_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---- Routes ----
@app.route("/api/permissions", methods=["GET"])
def api_permissions():
    """
    Read-only endpoint: returns the current contents of permissions.json
    as JSON (or [] if missing/invalid).
    """
    path = PERMISSIONS_FILE
    alt_path = "/data/permissions.json"

    data = []
    for p in (path, alt_path):
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                break
            except Exception:
                # invalid JSON -> laat data gewoon []
                data = []
                break

    return jsonify(data)

@app.route("/", methods=["GET", "POST"])
def index():
    config = load_config()
    worlds = list_worlds()
    error = None

    if request.method == "POST":
        form = request.form
        try:
            # GENERAL
            config["general"]["server_name"] = form.get("server_name", "").strip()
            config["general"]["server_port"] = to_int(
                form.get("server_port"), DEFAULT_CONFIG["general"]["server_port"]
            )
            config["general"]["server_port_v6"] = to_int(
                form.get("server_port_v6"), DEFAULT_CONFIG["general"]["server_port_v6"]
            )
            config["general"]["online_mode"] = to_bool(form.get("online_mode"))
            config["general"]["emit_server_telemetry"] = to_bool(
                form.get("emit_server_telemetry")
            )
            config["general"]["enable_lan_visibility"] = to_bool(
                form.get("enable_lan_visibility")
            )

            # WORLD
            selected_world = form.get("selected_world", "").strip()
            new_world_name = form.get("new_world_name", "").strip()

            # Seed, type, etc.
            config["world"]["level_seed"] = form.get("level_seed", "").strip()
            config["world"]["level_type"] = form.get(
                "level_type", DEFAULT_CONFIG["world"]["level_type"]
            )
            config["world"]["gamemode"] = form.get(
                "gamemode", DEFAULT_CONFIG["world"]["gamemode"]
            )
            config["world"]["difficulty"] = form.get(
                "difficulty", DEFAULT_CONFIG["world"]["difficulty"]
            )
            config["world"]["allow_cheats"] = to_bool(form.get("allow_cheats"))

            # World mapping:
            if new_world_name:
                # Nieuwe wereld: map aanmaken als nodig
                world_dir = os.path.join(WORLDS_DIR, new_world_name)
                if not os.path.exists(world_dir):
                    os.makedirs(world_dir, exist_ok=True)
                config["world"]["level_name"] = new_world_name
            elif selected_world:
                config["world"]["level_name"] = selected_world
            else:
                # Geen selectie, terugvallen op huidige config of default
                if not config["world"].get("level_name"):
                    config["world"]["level_name"] = DEFAULT_CONFIG["world"]["level_name"]

            # PLAYERS
            config["players"]["max_players"] = to_int(
                form.get("max_players"), DEFAULT_CONFIG["players"]["max_players"]
            )
            config["players"]["white_list"] = to_bool(form.get("white_list"))
            config["players"]["allow_list"] = to_bool(form.get("allow_list"))
            config["players"]["default_player_permission_level"] = form.get(
                "default_player_permission_level",
                DEFAULT_CONFIG["players"]["default_player_permission_level"],
            )
            config["players"]["texturepack_required"] = to_bool(
                form.get("texturepack_required")
            )

            # role_assignments als JSON textarea (optioneel)
            ra_raw = form.get("role_assignments_json", "").strip()
            if ra_raw:
                config["players"]["role_assignments"] = json.loads(ra_raw)
            else:
                config["players"]["role_assignments"] = []

            # PERFORMANCE
            config["performance"]["view_distance"] = to_int(
                form.get("view_distance"),
                DEFAULT_CONFIG["performance"]["view_distance"],
            )
            config["performance"]["tick_distance"] = to_int(
                form.get("tick_distance"),
                DEFAULT_CONFIG["performance"]["tick_distance"],
            )
            config["performance"]["player_idle_timeout"] = to_int(
                form.get("player_idle_timeout"),
                DEFAULT_CONFIG["performance"]["player_idle_timeout"],
            )
            config["performance"]["max_threads"] = to_int(
                form.get("max_threads"),
                DEFAULT_CONFIG["performance"]["max_threads"],
            )
            config["performance"]["compression_threshold"] = to_int(
                form.get("compression_threshold"),
                DEFAULT_CONFIG["performance"]["compression_threshold"],
            )

            # ANTI-CHEAT
            config["anti_cheat"]["server_authoritative_movement"] = form.get(
                "server_authoritative_movement",
                DEFAULT_CONFIG["anti_cheat"]["server_authoritative_movement"],
            )
            config["anti_cheat"]["server_authoritative_block_breaking"] = to_bool(
                form.get("server_authoritative_block_breaking")
            )
            config["anti_cheat"]["player_movement_score_threshold"] = to_int(
                form.get("player_movement_score_threshold"),
                DEFAULT_CONFIG["anti_cheat"]["player_movement_score_threshold"],
            )
            config["anti_cheat"]["player_movement_distance_threshold"] = to_float(
                form.get("player_movement_distance_threshold"),
                DEFAULT_CONFIG["anti_cheat"]["player_movement_distance_threshold"],
            )
            config["anti_cheat"][
                "player_movement_duration_threshold_in_ms"
            ] = to_int(
                form.get("player_movement_duration_threshold_in_ms"),
                DEFAULT_CONFIG["anti_cheat"][
                    "player_movement_duration_threshold_in_ms"
                ],
            )
            config["anti_cheat"]["correct_player_movement"] = to_bool(
                form.get("correct_player_movement")
            )

            save_config(config)
            flash("Configuration saved. Restart the add-on to apply changes.", "success")
            return redirect(url_for("index"))

        except Exception as exc:
            error = f"Error while saving configuration: {exc}"
            flash(error, "error")

    # Voor role_assignments textarea
    role_assignments_json = json.dumps(
        config["players"].get("role_assignments", []), indent=2
    )

    current_world = config["world"].get("level_name") or DEFAULT_CONFIG["world"][
        "level_name"
    ]
    # zorg dat huidige wereld in de lijst zit
    if current_world and current_world not in worlds:
        worlds.append(current_world)
        worlds.sort()

    return render_template_string(
        TEMPLATE,
        config=config,
        worlds=worlds,
        current_world=current_world,
        role_assignments_json=role_assignments_json,
    )


# ---- Template met Bootstrap 5.2.3 ----

TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Minecraft Server for HA</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap CSS -->
  <link href="static/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-dark text-light">
<nav class="navbar navbar-dark bg-dark border-bottom border-secondary mb-3">
  <div class="container-fluid">
    <span class="navbar-brand mb-0 h1">Minecraft Server for Home Assistant</span>
    <span class="text-muted small">Configure Bedrock server settings via Ingress</span>
  </div>
</nav>

<div class="container pb-5">
  {% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
      {% for category, msg in messages %}
        <div class="alert alert-{{ 'success' if category == 'success' else 'danger' }} alert-sm" role="alert">
          {{ msg }}
        </div>
      {% endfor %}
    {% endif %}
  {% endwith %}

  <form method="post" class="row g-3">
    <!-- General -->
    <div class="col-12 col-lg-6">
      <div class="card bg-dark border-secondary">
        <div class="card-header border-secondary">
          <strong>General</strong>
        </div>
        <div class="card-body">
          <div class="mb-3">
            <label for="server_name" class="form-label">Server name</label>
            <input type="text" class="form-control form-control-sm bg-black text-light" id="server_name" name="server_name"
                   value="{{ config.general.server_name }}">
          </div>
          <div class="mb-3">
            <label for="server_port" class="form-label">Server port (IPv4)</label>
            <input type="number" class="form-control form-control-sm bg-black text-light" id="server_port" name="server_port"
                   value="{{ config.general.server_port }}">
          </div>
          <div class="mb-3">
            <label for="server_port_v6" class="form-label">Server port (IPv6)</label>
            <input type="number" class="form-control form-control-sm bg-black text-light" id="server_port_v6" name="server_port_v6"
                   value="{{ config.general.server_port_v6 }}">
          </div>
          <div class="form-check form-switch mb-2">
            <input class="form-check-input" type="checkbox" id="online_mode" name="online_mode"
                   {% if config.general.online_mode %}checked{% endif %}>
            <label class="form-check-label" for="online_mode">Online mode (Xbox Live authentication)</label>
          </div>
          <div class="form-check form-switch mb-2">
            <input class="form-check-input" type="checkbox" id="emit_server_telemetry" name="emit_server_telemetry"
                   {% if config.general.emit_server_telemetry %}checked{% endif %}>
            <label class="form-check-label" for="emit_server_telemetry">Emit server telemetry</label>
          </div>
          <div class="form-check form-switch mb-2">
            <input class="form-check-input" type="checkbox" id="enable_lan_visibility" name="enable_lan_visibility"
                   {% if config.general.enable_lan_visibility %}checked{% endif %}>
            <label class="form-check-label" for="enable_lan_visibility">Visible on local network (LAN)</label>
          </div>
        </div>
      </div>
    </div>

    <!-- World -->
    <div class="col-12 col-lg-6">
      <div class="card bg-dark border-secondary">
        <div class="card-header border-secondary">
          <strong>World</strong>
        </div>
        <div class="card-body">
          <div class="mb-3">
            <label for="selected_world" class="form-label">Existing world</label>
            <select class="form-select form-select-sm bg-black text-light" id="selected_world" name="selected_world">
              <option value="">-- Select world --</option>
              {% for w in worlds %}
                <option value="{{ w }}" {% if current_world == w %}selected{% endif %}>{{ w }}</option>
              {% endfor %}
            </select>
            <div class="form-text text-muted">Select an existing world folder from /data/worlds.</div>
          </div>
          <div class="mb-3">
            <label for="new_world_name" class="form-label">New world name</label>
            <input type="text" class="form-control form-control-sm bg-black text-light" id="new_world_name" name="new_world_name"
                   placeholder="Leave empty to use selected world">
            <div class="form-text text-muted">
              If you enter a new world name, a folder will be created under <code>/data/worlds/&lt;name&gt;</code> and used as level name.
            </div>
          </div>
          <div class="mb-3">
            <label for="level_seed" class="form-label">Level seed</label>
            <input type="text" class="form-control form-control-sm bg-black text-light" id="level_seed" name="level_seed"
                   value="{{ config.world.level_seed }}">
            <div class="form-text text-muted">Leave empty for a random seed.</div>
          </div>
          <div class="mb-3">
            <label for="level_type" class="form-label">Level type</label>
            <select class="form-select form-select-sm bg-black text-light" id="level_type" name="level_type">
              {% for val,label in [("DEFAULT","Default"),("FLAT","Flat"),("LEGACY","Legacy")] %}
                <option value="{{ val }}" {% if config.world.level_type == val %}selected{% endif %}>{{ label }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="mb-3">
            <label for="gamemode" class="form-label">Gamemode</label>
            <select class="form-select form-select-sm bg-black text-light" id="gamemode" name="gamemode">
              {% for val,label in [("survival","Survival"),("creative","Creative"),("adventure","Adventure")] %}
                <option value="{{ val }}" {% if config.world.gamemode == val %}selected{% endif %}>{{ label }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="mb-3">
            <label for="difficulty" class="form-label">Difficulty</label>
            <select class="form-select form-select-sm bg-black text-light" id="difficulty" name="difficulty">
              {% for val,label in [("peaceful","Peaceful"),("easy","Easy"),("normal","Normal"),("hard","Hard")] %}
                <option value="{{ val }}" {% if config.world.difficulty == val %}selected{% endif %}>{{ label }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="form-check form-switch mb-2">
            <input class="form-check-input" type="checkbox" id="allow_cheats" name="allow_cheats"
                   {% if config.world.allow_cheats %}checked{% endif %}>
            <label class="form-check-label" for="allow_cheats">Allow cheats</label>
          </div>
        </div>
      </div>
    </div>

    <!-- Players -->
    <div class="col-12 col-lg-6">
      <div class="card bg-dark border-secondary">
        <div class="card-header border-secondary">
          <strong>Players</strong>
        </div>
        <div class="card-body">
          <div class="mb-3">
            <label for="max_players" class="form-label">Max players</label>
            <input type="number" class="form-control form-control-sm bg-black text-light" id="max_players" name="max_players"
                   value="{{ config.players.max_players }}">
          </div>
          <div class="form-check form-switch mb-2">
            <input class="form-check-input" type="checkbox" id="white_list" name="white_list"
                   {% if config.players.white_list %}checked{% endif %}>
            <label class="form-check-label" for="white_list">Whitelist enabled</label>
          </div>
          <div class="form-check form-switch mb-2">
            <input class="form-check-input" type="checkbox" id="allow_list" name="allow_list"
                   {% if config.players.allow_list %}checked{% endif %}>
            <label class="form-check-label" for="allow_list">Allow list enabled</label>
          </div>
          <div class="mb-3">
            <label for="default_player_permission_level" class="form-label">Default permission level</label>
            <select class="form-select form-select-sm bg-black text-light" id="default_player_permission_level" name="default_player_permission_level">
              {% for val,label in [("visitor","Visitor"),("member","Member"),("operator","Operator")] %}
                <option value="{{ val }}" {% if config.players.default_player_permission_level == val %}selected{% endif %}>{{ label }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="form-check form-switch mb-2">
            <input class="form-check-input" type="checkbox" id="texturepack_required" name="texturepack_required"
                   {% if config.players.texturepack_required %}checked{% endif %}>
            <label class="form-check-label" for="texturepack_required">Require texture pack</label>
          </div>
          <div class="mb-3">
            <label for="role_assignments_json" class="form-label">Role assignments (JSON)</label>
            <textarea class="form-control form-control-sm bg-black text-light" id="role_assignments_json" name="role_assignments_json">{{ role_assignments_json }}</textarea>
            <div class="form-text text-muted">
              Array of objects with <code>xuid</code> and <code>role</code>.
              Example: <code>[{"xuid":"123","role":"operator"}]</code>
            </div>
          </div>
          <div class="mb-3">
            <label class="form-label">Runtime permissions (permissions.json)</label>
            <pre id="runtime_permissions"
                class="form-control form-control-sm bg-black text-light"
                style="min-height: 120px; white-space: pre; overflow-x: auto;"></pre>
            <div class="form-text text-muted">
              Read-only view of the current <code>permissions.json</code> used by Bedrock.
              This may change during runtime (e.g. via <code>/op</code> commands).
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Performance -->
    <div class="col-12 col-lg-6">
      <div class="card bg-dark border-secondary">
        <div class="card-header border-secondary">
          <strong>Performance</strong>
        </div>
        <div class="card-body">
          <div class="mb-3">
            <label for="view_distance" class="form-label">View distance</label>
            <input type="number" class="form-control form-control-sm bg-black text-light" id="view_distance" name="view_distance"
                   value="{{ config.performance.view_distance }}">
          </div>
          <div class="mb-3">
            <label for="tick_distance" class="form-label">Tick distance</label>
            <input type="number" class="form-control form-control-sm bg-black text-light" id="tick_distance" name="tick_distance"
                   value="{{ config.performance.tick_distance }}">
          </div>
          <div class="mb-3">
            <label for="player_idle_timeout" class="form-label">Player idle timeout (minutes)</label>
            <input type="number" class="form-control form-control-sm bg-black text-light" id="player_idle_timeout" name="player_idle_timeout"
                   value="{{ config.performance.player_idle_timeout }}">
          </div>
          <div class="mb-3">
            <label for="max_threads" class="form-label">Max threads (0 = auto)</label>
            <input type="number" class="form-control form-control-sm bg-black text-light" id="max_threads" name="max_threads"
                   value="{{ config.performance.max_threads }}">
          </div>
          <div class="mb-3">
            <label for="compression_threshold" class="form-label">Compression threshold</label>
            <input type="number" class="form-control form-control-sm bg-black text-light" id="compression_threshold" name="compression_threshold"
                   value="{{ config.performance.compression_threshold }}">
          </div>
        </div>
      </div>
    </div>

    <!-- Anti-cheat -->
    <div class="col-12 col-lg-6">
      <div class="card bg-dark border-secondary">
        <div class="card-header border-secondary">
          <strong>Anti-cheat</strong>
        </div>
        <div class="card-body">
          <div class="mb-3">
            <label for="server_authoritative_movement" class="form-label">Authoritative movement</label>
            <select class="form-select form-select-sm bg-black text-light" id="server_authoritative_movement" name="server_authoritative_movement">
              {% for val,label in [("server-auth","Server auth"),("client-auth","Client auth"),("server-auth-with-rewind","Server auth with rewind")] %}
                <option value="{{ val }}" {% if config.anti_cheat.server_authoritative_movement == val %}selected{% endif %}>{{ label }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="form-check form-switch mb-2">
            <input class="form-check-input" type="checkbox" id="server_authoritative_block_breaking" name="server_authoritative_block_breaking"
                   {% if config.anti_cheat.server_authoritative_block_breaking %}checked{% endif %}>
            <label class="form-check-label" for="server_authoritative_block_breaking">Server authoritative block breaking</label>
          </div>
          <div class="mb-3">
            <label for="player_movement_score_threshold" class="form-label">Movement score threshold</label>
            <input type="number" class="form-control form-control-sm bg-black text-light" id="player_movement_score_threshold" name="player_movement_score_threshold"
                   value="{{ config.anti_cheat.player_movement_score_threshold }}">
          </div>
          <div class="mb-3">
            <label for="player_movement_distance_threshold" class="form-label">Movement distance threshold</label>
            <input type="number" step="0.01" class="form-control form-control-sm bg-black text-light" id="player_movement_distance_threshold" name="player_movement_distance_threshold"
                   value="{{ config.anti_cheat.player_movement_distance_threshold }}">
          </div>
          <div class="mb-3">
            <label for="player_movement_duration_threshold_in_ms" class="form-label">Movement duration threshold (ms)</label>
            <input type="number" class="form-control form-control-sm bg-black text-light" id="player_movement_duration_threshold_in_ms" name="player_movement_duration_threshold_in_ms"
                   value="{{ config.anti_cheat.player_movement_duration_threshold_in_ms }}">
          </div>
          <div class="form-check form-switch mb-2">
            <input class="form-check-input" type="checkbox" id="correct_player_movement" name="correct_player_movement"
                   {% if config.anti_cheat.correct_player_movement %}checked{% endif %}>
            <label class="form-check-label" for="correct_player_movement">Correct player movement</label>
          </div>
        </div>
      </div>
    </div>

    <div class="col-12 d-flex justify-content-end mt-2">
      <button type="submit" class="btn btn-success btn-sm px-4">Save configuration</button>
    </div>
  </form>
</div>

<!-- Bootstrap JS -->
  <script src="static/bootstrap.bundle.min.js"></script>
  <script>
  (function() {
    const el = document.getElementById('runtime_permissions');
    if (!el) return;

    fetch("{{ url_for('api_permissions') }}")
      .then(resp => resp.json())
      .then(data => {
        el.textContent = JSON.stringify(data, null, 2);
      })
      .catch(err => {
        el.textContent = "Error reading permissions.json: " + err;
      });
  })();
  </script>

</body>
</html>
"""

if __name__ == "__main__":
    # Handig voor lokale debug (buiten HA)
    app.run(host="0.0.0.0", port=8789, debug=True)
