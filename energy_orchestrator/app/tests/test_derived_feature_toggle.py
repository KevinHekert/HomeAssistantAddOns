"""
Test toggling derived features from sensor stats configuration.

This test reproduces the issue where toggling a derived feature like 'wind_avg_1h'
results in a 404 error because it doesn't exist in CORE_FEATURES or EXPERIMENTAL_FEATURES.
"""

import json
import logging
import pytest
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from app import app
from db.core import init_db_schema, engine
from db import Sample, ResampledSample
from db.sensor_category_config import SensorCategoryConfiguration, SensorDefinition, SensorType
from db.feature_stats import FeatureStatsConfiguration, StatType
from ml.feature_config import FeatureConfiguration

_Logger = logging.getLogger(__name__)


class TestDerivedFeatureToggle:
    """Test toggling derived features from sensor configuration."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create test client with temporary database."""
        # Set environment variables for test database
        with patch.dict(os.environ, {
            "CONFIG_DIR": str(tmp_path),
            "DATA_DIR": str(tmp_path),
            "DATABASE_URL": f"sqlite:///{tmp_path}/test.db"
        }):
            app.config['TESTING'] = True
            init_db_schema()
            
            with app.test_client() as client:
                yield client

    def test_toggle_derived_feature_from_sensor_stats(self, client, tmp_path):
        """
        Test toggling a derived feature (e.g., wind_avg_1h) that comes from sensor stats.
        
        The issue: When feature stats configuration has wind_avg_1h enabled,
        it shows up in sensor_cards, but toggling it fails with 404 because
        it's not in CORE_FEATURES or EXPERIMENTAL_FEATURES.
        
        This should succeed without 404 errors.
        """
        # Step 1: Enable wind sensor and configure stats
        sensor_config = SensorCategoryConfiguration()
        assert sensor_config.get_sensor_config("wind") is not None
        sensor_config.save()
        
        # Step 2: Create feature stats configuration with avg_1h enabled for wind
        stats_config = FeatureStatsConfiguration()
        stats_config.set_stat_enabled("wind", StatType.AVG_1H, True)
        stats_config.save()
        
        # Step 3: Verify that sensor cards endpoint works (cards show sensors)
        response = client.get("/api/features/sensor_cards")
        assert response.status_code == 200
        data = json.loads(response.data)
        
        wind_card = None
        for card in data["sensor_cards"]:
            if card["sensor_name"] == "wind":
                wind_card = card
                break
        
        assert wind_card is not None
        
        # Step 4: Try to toggle the wind_avg_1h feature (this is where the bug occurs)
        # This is the core test - toggling a derived feature should work
        response = client.post(
            "/api/features/toggle",
            json={"feature_name": "wind_avg_1h", "enabled": True}
        )
        
        # This should succeed, not return 404
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.data}"
        data = json.loads(response.data)
        assert data["status"] == "success"
        assert data["message"] == "Feature 'wind_avg_1h' is now enabled"
        assert "wind_avg_1h" in data["active_features"]
        
        # Step 5: Verify the feature is now in the config
        response = client.get("/api/features/config")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "wind_avg_1h" in data["active_feature_names"]
        
        # Step 6: Disable it again
        response = client.post(
            "/api/features/toggle",
            json={"feature_name": "wind_avg_1h", "enabled": False}
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "success"
        assert "wind_avg_1h" not in data["active_features"]

    def test_toggle_nonexistent_derived_feature(self, client, tmp_path):
        """
        Test toggling a feature that looks like a derived feature but doesn't correspond
        to any configured sensor stat.
        """
        # Try to toggle a feature that doesn't exist anywhere
        response = client.post(
            "/api/features/toggle",
            json={"feature_name": "nonexistent_sensor_avg_1h", "enabled": True}
        )
        
        # This should return 404 as the feature truly doesn't exist
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data["status"] == "error"
        assert "not found" in data["message"].lower()

    def test_sensor_cards_show_derived_features(self, client, tmp_path):
        """
        Test that sensor cards correctly show derived features from sensor stats config.
        
        This test verifies that when a derived feature is enabled via toggle,
        it appears in the sensor cards endpoint.
        """
        # Enable wind sensor
        sensor_config = SensorCategoryConfiguration()
        assert sensor_config.get_sensor_config("wind") is not None
        
        # Enable the derived feature wind_avg_1h via toggle
        response = client.post(
            "/api/features/toggle",
            json={"feature_name": "wind_avg_1h", "enabled": True}
        )
        assert response.status_code == 200
        
        # Get sensor cards
        response = client.get("/api/features/sensor_cards")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "success"
        
        # Find wind sensor card
        wind_card = None
        for card in data["sensor_cards"]:
            if card["sensor_name"] == "wind":
                wind_card = card
                break
        
        assert wind_card is not None, "Wind sensor card should exist"
        
        # The wind feature should be shown
        feature_names = [f["name"] for f in wind_card["features"]]
        assert "wind" in feature_names
        
        # Verify wind_avg_1h is in the active features
        response = client.get("/api/features/config")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "wind_avg_1h" in data["active_feature_names"]
