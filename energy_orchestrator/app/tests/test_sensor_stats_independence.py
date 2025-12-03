"""
Test that sensor statistics configuration remains independent from ML feature configuration.

This test reproduces the bug described in the issue:
- Configure sensor stats (e.g., wind with avg_1h, 6h, 24h, 7d)
- Toggle a feature card (enable/disable for model training)
- Sensor stats configuration should NOT be affected
- Only the ML feature config should change
"""

import json
import logging
import pytest
import os
from pathlib import Path
from unittest.mock import patch

from app import app
from db.core import init_db_schema
from db.sensor_category_config import SensorCategoryConfiguration
from db.feature_stats import FeatureStatsConfiguration, StatType
from ml.feature_config import FeatureConfiguration

_Logger = logging.getLogger(__name__)


class TestSensorStatsIndependence:
    """Test that sensor stats configuration is independent from ML feature toggling."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create test client with temporary database."""
        with patch.dict(os.environ, {
            "CONFIG_DIR": str(tmp_path),
            "DATA_DIR": str(tmp_path),
            "DATABASE_URL": f"sqlite:///{tmp_path}/test.db"
        }):
            app.config['TESTING'] = True
            init_db_schema()
            
            # Clear global configs
            import db.feature_stats
            import ml.feature_config
            db.feature_stats._config = None
            ml.feature_config._config = None
            
            with app.test_client() as client:
                yield client

    def test_sensor_stats_persist_after_feature_toggle(self, client, tmp_path):
        """
        Test that sensor stats configuration persists when toggling ML features.
        
        This is the core issue:
        1. Configure wind sensor with all stats (avg_1h, 6h, 24h, 7d)
        2. Toggle wind_avg_1h feature for model training
        3. Sensor stats configuration should remain unchanged
        4. Only the ML feature config should change
        """
        # Step 1: Enable wind sensor
        sensor_config = SensorCategoryConfiguration()
        wind_sensor = sensor_config.get_sensor_config("wind")
        assert wind_sensor is not None
        sensor_config.save()
        
        # Step 2: Configure wind sensor with ALL stats enabled
        stats_config = FeatureStatsConfiguration()
        stats_config.set_stat_enabled("wind", StatType.AVG_1H, True)
        stats_config.set_stat_enabled("wind", StatType.AVG_6H, True)
        stats_config.set_stat_enabled("wind", StatType.AVG_24H, True)
        stats_config.set_stat_enabled("wind", StatType.AVG_7D, True)
        stats_config.save()
        
        # Verify all stats are enabled
        wind_stats = stats_config.get_enabled_stats_for_sensor("wind")
        assert StatType.AVG_1H in wind_stats
        assert StatType.AVG_6H in wind_stats
        assert StatType.AVG_24H in wind_stats
        assert StatType.AVG_7D in wind_stats
        assert len(wind_stats) == 4, "All 4 stats should be enabled"
        
        # Step 3: Toggle wind_avg_1h feature for ML training
        response = client.post(
            "/api/features/toggle",
            json={"feature_name": "wind_avg_1h", "enabled": True}
        )
        assert response.status_code == 200, f"Toggle failed: {response.data}"
        
        # Step 4: Verify sensor stats configuration is UNCHANGED
        # Reload from disk to ensure we're checking persisted state
        from db.feature_stats import reload_feature_stats_config
        stats_config_after = reload_feature_stats_config()
        wind_stats_after = stats_config_after.get_enabled_stats_for_sensor("wind")
        
        # BUG: These assertions currently fail because sync_stats_config_with_features()
        # overwrites the sensor stats configuration
        assert StatType.AVG_1H in wind_stats_after, "AVG_1H should still be enabled"
        assert StatType.AVG_6H in wind_stats_after, "AVG_6H should still be enabled"
        assert StatType.AVG_24H in wind_stats_after, "AVG_24H should still be enabled"
        assert StatType.AVG_7D in wind_stats_after, "AVG_7D should still be enabled"
        assert len(wind_stats_after) == 4, "All 4 stats should still be enabled"
        
        # Step 5: Verify ML feature config WAS changed
        response = client.get("/api/features/config")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "wind_avg_1h" in data["active_feature_names"], "Feature should be enabled"

    def test_indoor_temp_stats_persist_after_toggle(self, client, tmp_path):
        """
        Test with indoor_temp sensor to ensure the fix works for different sensors.
        """
        # Step 1: Enable indoor_temp sensor
        sensor_config = SensorCategoryConfiguration()
        indoor_sensor = sensor_config.get_sensor_config("indoor_temp")
        assert indoor_sensor is not None
        sensor_config.save()
        
        # Step 2: Configure indoor_temp with all stats
        stats_config = FeatureStatsConfiguration()
        stats_config.set_stat_enabled("indoor_temp", StatType.AVG_1H, True)
        stats_config.set_stat_enabled("indoor_temp", StatType.AVG_6H, True)
        stats_config.set_stat_enabled("indoor_temp", StatType.AVG_24H, True)
        stats_config.set_stat_enabled("indoor_temp", StatType.AVG_7D, True)
        stats_config.save()
        
        # Verify all stats are enabled
        indoor_stats = stats_config.get_enabled_stats_for_sensor("indoor_temp")
        assert len(indoor_stats) == 4
        
        # Step 3: Toggle a different feature (wind_avg_1h)
        # This should NOT affect indoor_temp stats
        response = client.post(
            "/api/features/toggle",
            json={"feature_name": "wind_avg_1h", "enabled": True}
        )
        assert response.status_code == 200
        
        # Step 4: Verify indoor_temp stats are UNCHANGED
        from db.feature_stats import reload_feature_stats_config
        stats_config_after = reload_feature_stats_config()
        indoor_stats_after = stats_config_after.get_enabled_stats_for_sensor("indoor_temp")
        
        assert StatType.AVG_1H in indoor_stats_after
        assert StatType.AVG_6H in indoor_stats_after
        assert StatType.AVG_24H in indoor_stats_after
        assert StatType.AVG_7D in indoor_stats_after
        assert len(indoor_stats_after) == 4, "All indoor_temp stats should be unchanged"

    def test_multiple_toggles_preserve_sensor_stats(self, client, tmp_path):
        """
        Test that multiple feature toggles don't corrupt sensor stats configuration.
        """
        # Configure wind sensor with specific stats
        stats_config = FeatureStatsConfiguration()
        stats_config.set_stat_enabled("wind", StatType.AVG_1H, True)
        stats_config.set_stat_enabled("wind", StatType.AVG_6H, True)
        stats_config.save()
        
        # Toggle multiple features
        client.post("/api/features/toggle", json={"feature_name": "wind_avg_1h", "enabled": True})
        client.post("/api/features/toggle", json={"feature_name": "wind_avg_1h", "enabled": False})
        client.post("/api/features/toggle", json={"feature_name": "wind_avg_1h", "enabled": True})
        
        # Verify sensor stats remain unchanged
        from db.feature_stats import reload_feature_stats_config
        stats_config_after = reload_feature_stats_config()
        wind_stats = stats_config_after.get_enabled_stats_for_sensor("wind")
        
        assert StatType.AVG_1H in wind_stats
        assert StatType.AVG_6H in wind_stats
        assert len(wind_stats) == 2, "Both configured stats should still be enabled"
