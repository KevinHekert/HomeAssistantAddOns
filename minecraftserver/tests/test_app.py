import json

import pytest

from minecraftserver.web import app


@pytest.fixture
def temp_paths(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    worlds_dir = tmp_path / "worlds"
    runtime_dir = tmp_path / "run"
    config_file = config_dir / "bedrock_for_ha_config.json"

    app.configure_data_dir(str(tmp_path))
    monkeypatch.setattr(app, "CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(app, "WORLDS_DIR", str(worlds_dir))
    monkeypatch.setattr(app, "CONFIG_FILE", str(config_file))
    monkeypatch.setattr(app, "RUNTIME_DIR", str(runtime_dir))

    return config_dir, worlds_dir, config_file


def test_deep_merge_nested_overrides():
    defaults = {"a": 1, "nested": {"one": 1, "two": 2}}
    overrides = {"nested": {"two": 22, "three": 3}, "b": 2}

    merged = app.deep_merge(defaults, overrides)

    assert merged == {
        "a": 1,
        "b": 2,
        "nested": {"one": 1, "two": 22, "three": 3},
    }
    assert defaults == {"a": 1, "nested": {"one": 1, "two": 2}}


def test_coercion_helpers():
    assert app.to_bool("true") is True
    assert app.to_bool("off") is False
    assert app.to_bool(None) is False

    assert app.to_int("10") == 10
    assert app.to_int("not-a-number", default=5) == 5

    assert app.to_float("1.5") == 1.5
    assert app.to_float("oops", default=2.5) == 2.5


def test_ensure_and_load_config_creates_defaults(temp_paths):
    _, _, config_file = temp_paths

    app.ensure_config_file()

    assert config_file.exists()
    with open(config_file, "r", encoding="utf-8") as file:
        stored = json.load(file)
    assert stored == app.DEFAULT_CONFIG

    loaded = app.load_config()
    assert loaded == app.DEFAULT_CONFIG


def test_load_config_merges_existing_values(temp_paths):
    _, _, config_file = temp_paths
    config_file.parent.mkdir(parents=True, exist_ok=True)
    partial_config = {
        "general": {"server_name": "MyServer"},
        "world": {"level_seed": "123"},
    }
    with open(config_file, "w", encoding="utf-8") as file:
        json.dump(partial_config, file)

    merged = app.load_config()

    assert merged["general"]["server_name"] == "MyServer"
    assert merged["world"]["level_seed"] == "123"
    assert merged["general"]["server_port"] == app.DEFAULT_CONFIG["general"]["server_port"]
    assert merged["players"] == app.DEFAULT_CONFIG["players"]


def test_save_config_writes_file(temp_paths):
    _, _, config_file = temp_paths
    data = {"general": {"server_name": "Saved"}}

    app.save_config(data)

    with open(config_file, "r", encoding="utf-8") as file:
        saved = json.load(file)
    assert saved == data


def test_list_worlds_lists_only_directories(temp_paths):
    _, worlds_dir, _ = temp_paths
    (worlds_dir / "WorldA").mkdir(parents=True)
    (worlds_dir / "WorldB").mkdir(parents=True)
    (worlds_dir / "not_a_dir.txt").write_text("file")
    (worlds_dir / ".hidden").mkdir()

    worlds = app.list_worlds()

    assert worlds == ["WorldA", "WorldB"]


def test_api_permissions_returns_file_contents(tmp_path, monkeypatch):
    perm_file = tmp_path / "permissions.json"
    data = [{"xuid": "123", "permission": "operator"}]
    perm_file.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(app, "PERMISSIONS_FILE", str(perm_file))

    with app.app.test_client() as client:
        response = client.get("/api/permissions")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"] == data
    assert payload["path"] == str(perm_file)


def test_api_permissions_handles_invalid_json(tmp_path, monkeypatch):
    perm_file = tmp_path / "permissions.json"
    perm_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(app, "PERMISSIONS_FILE", str(perm_file))

    with app.app.test_client() as client:
        response = client.get("/api/permissions")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is False
    assert payload["data"] == []
    assert "Invalid JSON" in payload["error"]
