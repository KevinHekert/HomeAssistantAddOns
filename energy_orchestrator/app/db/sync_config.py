"""
Sync configuration management for sensor data synchronization.

This module provides functionality to manage configurable sync settings:
- backfill_days: Number of days to look back when no samples exist (default: 14)
- sync_window_days: Size of each sync window in days (default: 1)
- sensor_sync_interval: Wait time in seconds between syncing individual sensors (default: 1)
- sensor_loop_interval: Wait time in seconds between sync loop iterations (default: 1)

Configuration is stored in a JSON file at /data/sync_config.json (persistent storage).
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

_Logger = logging.getLogger(__name__)

# Configuration file path for persistent sync config storage
# In Home Assistant add-ons, /data is the persistent data directory
SYNC_CONFIG_FILE_PATH = Path(os.environ.get("DATA_DIR", "/data")) / "sync_config.json"

# Default values
DEFAULT_BACKFILL_DAYS = 14
DEFAULT_SYNC_WINDOW_DAYS = 1
DEFAULT_SENSOR_SYNC_INTERVAL = 1
DEFAULT_SENSOR_LOOP_INTERVAL = 1

# Valid ranges for configuration values
MIN_BACKFILL_DAYS = 1
MAX_BACKFILL_DAYS = 365
MIN_SYNC_WINDOW_DAYS = 1
MAX_SYNC_WINDOW_DAYS = 30
MIN_SYNC_INTERVAL = 1
MAX_SYNC_INTERVAL = 3600


@dataclass
class SyncConfig:
    """Configuration for sensor data synchronization."""

    backfill_days: int = DEFAULT_BACKFILL_DAYS
    sync_window_days: int = DEFAULT_SYNC_WINDOW_DAYS
    sensor_sync_interval: int = DEFAULT_SENSOR_SYNC_INTERVAL
    sensor_loop_interval: int = DEFAULT_SENSOR_LOOP_INTERVAL


def _load_sync_config() -> SyncConfig:
    """Load sync configuration from persistent file.

    Returns:
        SyncConfig with values from file or defaults if not configured.
    """
    config = SyncConfig()

    try:
        if SYNC_CONFIG_FILE_PATH.exists():
            with open(SYNC_CONFIG_FILE_PATH, "r") as f:
                data = json.load(f)

            if "backfill_days" in data:
                val = data["backfill_days"]
                if isinstance(val, int) and MIN_BACKFILL_DAYS <= val <= MAX_BACKFILL_DAYS:
                    config.backfill_days = val

            if "sync_window_days" in data:
                val = data["sync_window_days"]
                if isinstance(val, int) and MIN_SYNC_WINDOW_DAYS <= val <= MAX_SYNC_WINDOW_DAYS:
                    config.sync_window_days = val

            if "sensor_sync_interval" in data:
                val = data["sensor_sync_interval"]
                if isinstance(val, int) and MIN_SYNC_INTERVAL <= val <= MAX_SYNC_INTERVAL:
                    config.sensor_sync_interval = val

            if "sensor_loop_interval" in data:
                val = data["sensor_loop_interval"]
                if isinstance(val, int) and MIN_SYNC_INTERVAL <= val <= MAX_SYNC_INTERVAL:
                    config.sensor_loop_interval = val

    except (json.JSONDecodeError, OSError) as e:
        _Logger.warning("Error loading sync config: %s. Using defaults.", e)

    return config


def _save_sync_config(config: SyncConfig) -> bool:
    """Save sync configuration to persistent file.

    Args:
        config: SyncConfig to save.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        # Ensure parent directory exists
        SYNC_CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "backfill_days": config.backfill_days,
            "sync_window_days": config.sync_window_days,
            "sensor_sync_interval": config.sensor_sync_interval,
            "sensor_loop_interval": config.sensor_loop_interval,
        }

        with open(SYNC_CONFIG_FILE_PATH, "w") as f:
            json.dump(data, f, indent=2)

        _Logger.info("Sync config saved: %s", data)
        return True

    except OSError as e:
        _Logger.error("Error saving sync config: %s", e)
        return False


def get_sync_config() -> SyncConfig:
    """Get the current sync configuration.

    Returns:
        SyncConfig with current values.
    """
    return _load_sync_config()


def get_backfill_days() -> int:
    """Get the number of days to backfill when no samples exist.

    Returns:
        Number of days (default: 14).
    """
    return _load_sync_config().backfill_days


def get_sync_window_days() -> int:
    """Get the size of each sync window in days.

    Returns:
        Number of days per sync window (default: 1).
    """
    return _load_sync_config().sync_window_days


def get_sensor_sync_interval() -> int:
    """Get the wait time in seconds between syncing individual sensors.

    Returns:
        Interval in seconds (default: 1).
    """
    return _load_sync_config().sensor_sync_interval


def get_sensor_loop_interval() -> int:
    """Get the wait time in seconds between sync loop iterations.

    Returns:
        Interval in seconds (default: 1).
    """
    return _load_sync_config().sensor_loop_interval


def set_sync_config(
    backfill_days: int | None = None,
    sync_window_days: int | None = None,
    sensor_sync_interval: int | None = None,
    sensor_loop_interval: int | None = None,
) -> tuple[bool, str | None]:
    """Update sync configuration values.

    Only provided (non-None) values are updated.

    Args:
        backfill_days: Number of days to backfill (1-365).
        sync_window_days: Size of sync window in days (1-30).
        sensor_sync_interval: Wait between sensors in seconds (1-3600).
        sensor_loop_interval: Wait between loop iterations in seconds (1-3600).

    Returns:
        Tuple of (success, error_message).
        error_message is None if successful.
    """
    config = _load_sync_config()

    if backfill_days is not None:
        if not isinstance(backfill_days, int) or not (MIN_BACKFILL_DAYS <= backfill_days <= MAX_BACKFILL_DAYS):
            return False, f"backfill_days must be between {MIN_BACKFILL_DAYS} and {MAX_BACKFILL_DAYS}"
        config.backfill_days = backfill_days

    if sync_window_days is not None:
        if not isinstance(sync_window_days, int) or not (MIN_SYNC_WINDOW_DAYS <= sync_window_days <= MAX_SYNC_WINDOW_DAYS):
            return False, f"sync_window_days must be between {MIN_SYNC_WINDOW_DAYS} and {MAX_SYNC_WINDOW_DAYS}"
        config.sync_window_days = sync_window_days

    if sensor_sync_interval is not None:
        if not isinstance(sensor_sync_interval, int) or not (MIN_SYNC_INTERVAL <= sensor_sync_interval <= MAX_SYNC_INTERVAL):
            return False, f"sensor_sync_interval must be between {MIN_SYNC_INTERVAL} and {MAX_SYNC_INTERVAL}"
        config.sensor_sync_interval = sensor_sync_interval

    if sensor_loop_interval is not None:
        if not isinstance(sensor_loop_interval, int) or not (MIN_SYNC_INTERVAL <= sensor_loop_interval <= MAX_SYNC_INTERVAL):
            return False, f"sensor_loop_interval must be between {MIN_SYNC_INTERVAL} and {MAX_SYNC_INTERVAL}"
        config.sensor_loop_interval = sensor_loop_interval

    if _save_sync_config(config):
        return True, None
    return False, "Failed to save configuration"
