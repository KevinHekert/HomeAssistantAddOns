"""
Tests for the sync configuration module.

Tests the get/set functionality for sync configuration settings.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from db.sync_config import (
    SyncConfig,
    get_sync_config,
    set_sync_config,
    get_backfill_days,
    get_sync_window_days,
    get_sensor_sync_interval,
    get_sensor_loop_interval,
    DEFAULT_BACKFILL_DAYS,
    DEFAULT_SYNC_WINDOW_DAYS,
    DEFAULT_SENSOR_SYNC_INTERVAL,
    DEFAULT_SENSOR_LOOP_INTERVAL,
    MIN_BACKFILL_DAYS,
    MAX_BACKFILL_DAYS,
    MIN_SYNC_WINDOW_DAYS,
    MAX_SYNC_WINDOW_DAYS,
    MIN_SYNC_INTERVAL,
    MAX_SYNC_INTERVAL,
)


class TestSyncConfigDefaults:
    """Tests for default values."""

    def test_default_backfill_days(self):
        """Default backfill days should be 14."""
        assert DEFAULT_BACKFILL_DAYS == 14

    def test_default_sync_window_days(self):
        """Default sync window days should be 1."""
        assert DEFAULT_SYNC_WINDOW_DAYS == 1

    def test_default_sensor_sync_interval(self):
        """Default sensor sync interval should be 1."""
        assert DEFAULT_SENSOR_SYNC_INTERVAL == 1

    def test_default_sensor_loop_interval(self):
        """Default sensor loop interval should be 1."""
        assert DEFAULT_SENSOR_LOOP_INTERVAL == 1


class TestSyncConfigLimits:
    """Tests for configuration limits."""

    def test_backfill_days_limits(self):
        """Backfill days limits should be 1-365."""
        assert MIN_BACKFILL_DAYS == 1
        assert MAX_BACKFILL_DAYS == 365

    def test_sync_window_days_limits(self):
        """Sync window days limits should be 1-30."""
        assert MIN_SYNC_WINDOW_DAYS == 1
        assert MAX_SYNC_WINDOW_DAYS == 30

    def test_sync_interval_limits(self):
        """Sync interval limits should be 1-3600."""
        assert MIN_SYNC_INTERVAL == 1
        assert MAX_SYNC_INTERVAL == 3600


class TestSyncConfigDataclass:
    """Tests for the SyncConfig dataclass."""

    def test_default_values(self):
        """SyncConfig should have correct default values."""
        config = SyncConfig()
        assert config.backfill_days == DEFAULT_BACKFILL_DAYS
        assert config.sync_window_days == DEFAULT_SYNC_WINDOW_DAYS
        assert config.sensor_sync_interval == DEFAULT_SENSOR_SYNC_INTERVAL
        assert config.sensor_loop_interval == DEFAULT_SENSOR_LOOP_INTERVAL

    def test_custom_values(self):
        """SyncConfig should accept custom values."""
        config = SyncConfig(
            backfill_days=30,
            sync_window_days=7,
            sensor_sync_interval=5,
            sensor_loop_interval=10,
        )
        assert config.backfill_days == 30
        assert config.sync_window_days == 7
        assert config.sensor_sync_interval == 5
        assert config.sensor_loop_interval == 10


class TestGetSyncConfig:
    """Tests for the get_sync_config function."""

    def test_returns_defaults_when_file_missing(self, tmp_path):
        """Should return defaults when config file doesn't exist."""
        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", tmp_path / "nonexistent.json"):
            config = get_sync_config()
            assert config.backfill_days == DEFAULT_BACKFILL_DAYS
            assert config.sync_window_days == DEFAULT_SYNC_WINDOW_DAYS

    def test_loads_values_from_file(self, tmp_path):
        """Should load values from config file."""
        config_file = tmp_path / "sync_config.json"
        config_file.write_text(json.dumps({
            "backfill_days": 30,
            "sync_window_days": 7,
            "sensor_sync_interval": 5,
            "sensor_loop_interval": 10,
        }))

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            config = get_sync_config()
            assert config.backfill_days == 30
            assert config.sync_window_days == 7
            assert config.sensor_sync_interval == 5
            assert config.sensor_loop_interval == 10

    def test_ignores_invalid_values(self, tmp_path):
        """Should ignore invalid values and use defaults."""
        config_file = tmp_path / "sync_config.json"
        config_file.write_text(json.dumps({
            "backfill_days": 1000,  # Over max
            "sync_window_days": 0,  # Under min
            "sensor_sync_interval": "invalid",  # Wrong type
        }))

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            config = get_sync_config()
            assert config.backfill_days == DEFAULT_BACKFILL_DAYS
            assert config.sync_window_days == DEFAULT_SYNC_WINDOW_DAYS
            assert config.sensor_sync_interval == DEFAULT_SENSOR_SYNC_INTERVAL

    def test_handles_corrupted_json(self, tmp_path):
        """Should return defaults for corrupted JSON."""
        config_file = tmp_path / "sync_config.json"
        config_file.write_text("not valid json {{{")

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            config = get_sync_config()
            assert config.backfill_days == DEFAULT_BACKFILL_DAYS


class TestSetSyncConfig:
    """Tests for the set_sync_config function."""

    def test_saves_valid_values(self, tmp_path):
        """Should save valid values to config file."""
        config_file = tmp_path / "sync_config.json"

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            success, error = set_sync_config(
                backfill_days=30,
                sync_window_days=7,
            )

            assert success is True
            assert error is None

            # Verify file contents
            saved = json.loads(config_file.read_text())
            assert saved["backfill_days"] == 30
            assert saved["sync_window_days"] == 7

    def test_rejects_invalid_backfill_days(self, tmp_path):
        """Should reject backfill_days outside valid range."""
        config_file = tmp_path / "sync_config.json"

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            success, error = set_sync_config(backfill_days=0)
            assert success is False
            assert "backfill_days" in error

            success, error = set_sync_config(backfill_days=400)
            assert success is False
            assert "backfill_days" in error

    def test_rejects_invalid_sync_window_days(self, tmp_path):
        """Should reject sync_window_days outside valid range."""
        config_file = tmp_path / "sync_config.json"

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            success, error = set_sync_config(sync_window_days=0)
            assert success is False
            assert "sync_window_days" in error

            success, error = set_sync_config(sync_window_days=50)
            assert success is False
            assert "sync_window_days" in error

    def test_rejects_invalid_sensor_sync_interval(self, tmp_path):
        """Should reject sensor_sync_interval outside valid range."""
        config_file = tmp_path / "sync_config.json"

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            success, error = set_sync_config(sensor_sync_interval=0)
            assert success is False
            assert "sensor_sync_interval" in error

            success, error = set_sync_config(sensor_sync_interval=5000)
            assert success is False
            assert "sensor_sync_interval" in error

    def test_rejects_invalid_sensor_loop_interval(self, tmp_path):
        """Should reject sensor_loop_interval outside valid range."""
        config_file = tmp_path / "sync_config.json"

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            success, error = set_sync_config(sensor_loop_interval=0)
            assert success is False
            assert "sensor_loop_interval" in error

            success, error = set_sync_config(sensor_loop_interval=5000)
            assert success is False
            assert "sensor_loop_interval" in error

    def test_partial_update(self, tmp_path):
        """Should only update provided values."""
        config_file = tmp_path / "sync_config.json"

        # First save initial values
        config_file.write_text(json.dumps({
            "backfill_days": 30,
            "sync_window_days": 7,
            "sensor_sync_interval": 5,
            "sensor_loop_interval": 10,
        }))

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            # Update only backfill_days
            success, error = set_sync_config(backfill_days=60)
            assert success is True

            # Verify all values are preserved
            config = get_sync_config()
            assert config.backfill_days == 60  # Updated
            assert config.sync_window_days == 7  # Preserved
            assert config.sensor_sync_interval == 5  # Preserved
            assert config.sensor_loop_interval == 10  # Preserved


class TestIndividualGetters:
    """Tests for individual getter functions."""

    def test_get_backfill_days(self, tmp_path):
        """Should return backfill_days from config."""
        config_file = tmp_path / "sync_config.json"
        config_file.write_text(json.dumps({"backfill_days": 30}))

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            assert get_backfill_days() == 30

    def test_get_sync_window_days(self, tmp_path):
        """Should return sync_window_days from config."""
        config_file = tmp_path / "sync_config.json"
        config_file.write_text(json.dumps({"sync_window_days": 7}))

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            assert get_sync_window_days() == 7

    def test_get_sensor_sync_interval(self, tmp_path):
        """Should return sensor_sync_interval from config."""
        config_file = tmp_path / "sync_config.json"
        config_file.write_text(json.dumps({"sensor_sync_interval": 5}))

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            assert get_sensor_sync_interval() == 5

    def test_get_sensor_loop_interval(self, tmp_path):
        """Should return sensor_loop_interval from config."""
        config_file = tmp_path / "sync_config.json"
        config_file.write_text(json.dumps({"sensor_loop_interval": 10}))

        with patch("db.sync_config.SYNC_CONFIG_FILE_PATH", config_file):
            assert get_sensor_loop_interval() == 10
