"""
Tests for Flask app endpoints.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd

from app import app
from db.resample import ResampleStats
from ml.heating_features import FeatureDatasetStats, TrainingDataRange


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_model():
    """Create a mock heating demand model."""
    model = MagicMock()
    model.is_available = True
    model.feature_names = ["outdoor_temp", "wind", "humidity"]
    model.training_timestamp = datetime(2024, 1, 1, 12, 0, 0)
    model.predict.return_value = 1.5
    return model


class TestResampleEndpoint:
    """Test the /resample POST endpoint (now runs in background)."""

    def test_resample_success(self, client):
        """Successful resample starts background thread and returns immediately."""
        with patch("app._run_resample_in_thread") as mock_thread_func:
            response = client.post("/resample")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "background" in data["message"].lower()
            assert data["running"] is True

    def test_resample_error(self, client):
        """Error during resample initialization returns 500 with error message."""
        # Simulate error in thread creation
        with patch("app.threading.Thread") as mock_thread:
            mock_thread.side_effect = Exception("Thread creation failed")

            response = client.post("/resample")

            assert response.status_code == 500
            data = response.get_json()
            assert data["status"] == "error"
            assert "Thread creation failed" in data["message"]


class TestIndexEndpoint:
    """Test the / GET endpoint."""

    def test_index_returns_html(self, client):
        """Index returns HTML page."""
        with patch("app.get_entity_state") as mock_state:
            mock_state.return_value = (10.5, "m/s")

            response = client.get("/")

            assert response.status_code == 200
            assert b"Energy Orchestrator" in response.data
            assert b"Resample Data" in response.data


class TestTrainHeatingDemandEndpoint:
    """Test the /api/train/heating_demand POST endpoint."""

    def test_train_success(self, client):
        """Successful training returns 200 with metrics."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = FeatureDatasetStats(
            total_slots=100,
            valid_slots=80,
            dropped_missing_features=10,
            dropped_missing_target=5,
            dropped_insufficient_history=5,
            features_used=["outdoor_temp", "wind"],
            has_7d_features=False,
        )
        mock_model = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.train_samples = 60
        mock_metrics.val_samples = 20
        mock_metrics.train_mae = 0.1
        mock_metrics.val_mae = 0.15
        mock_metrics.val_mape = 0.05
        mock_metrics.val_r2 = 0.85
        mock_metrics.features = ["outdoor_temp", "wind"]

        with patch("app.build_heating_feature_dataset") as mock_build, \
             patch("app.train_heating_demand_model") as mock_train:
            mock_build.return_value = (mock_df, mock_stats)
            mock_train.return_value = (mock_model, mock_metrics)

            response = client.post("/api/train/heating_demand")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "metrics" in data
            assert data["metrics"]["train_samples"] == 60
            assert data["metrics"]["val_samples"] == 20

    def test_train_insufficient_data(self, client):
        """Training with insufficient data returns 400."""
        mock_stats = FeatureDatasetStats(
            total_slots=30,
            valid_slots=10,
            dropped_missing_features=10,
            dropped_missing_target=5,
            dropped_insufficient_history=5,
            features_used=[],
            has_7d_features=False,
        )

        with patch("app.build_heating_feature_dataset") as mock_build:
            mock_build.return_value = (None, mock_stats)

            response = client.post("/api/train/heating_demand")

            assert response.status_code == 400
            data = response.get_json()
            assert data["status"] == "error"
            assert "Insufficient data" in data["message"]

    def test_train_error(self, client):
        """Error during training returns 500."""
        with patch("app.build_heating_feature_dataset") as mock_build:
            mock_build.side_effect = Exception("Training failed")

            response = client.post("/api/train/heating_demand")

            assert response.status_code == 500
            data = response.get_json()
            assert data["status"] == "error"

    def test_train_returns_training_data_range(self, client):
        """Successful training returns training_data with all sensor categories including hp_kwh_delta."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = FeatureDatasetStats(
            total_slots=100,
            valid_slots=80,
            dropped_missing_features=10,
            dropped_missing_target=5,
            dropped_insufficient_history=5,
            features_used=["outdoor_temp", "wind"],
            has_7d_features=False,
            sensor_ranges={
                "dhw_temp": TrainingDataRange(first=45.0, last=50.0, unit="°C"),
                "hp_kwh_total": TrainingDataRange(first=1000.0, last=1500.0, unit="kWh"),
                "outdoor_temp": TrainingDataRange(first=5.0, last=10.0, unit="°C"),
            },
            hp_kwh_delta=500.0,
            dhw_temp_range=TrainingDataRange(first=45.0, last=50.0, unit="°C"),
            hp_kwh_total_range=TrainingDataRange(first=1000.0, last=1500.0, unit="kWh"),
        )
        mock_model = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.train_samples = 60
        mock_metrics.val_samples = 20
        mock_metrics.train_mae = 0.1
        mock_metrics.val_mae = 0.15
        mock_metrics.val_mape = 0.05
        mock_metrics.val_r2 = 0.85
        mock_metrics.features = ["outdoor_temp", "wind"]

        with patch("app.build_heating_feature_dataset") as mock_build, \
             patch("app.train_heating_demand_model") as mock_train:
            mock_build.return_value = (mock_df, mock_stats)
            mock_train.return_value = (mock_model, mock_metrics)

            response = client.post("/api/train/heating_demand")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "training_data" in data
            # Check all sensor categories are present
            assert data["training_data"]["dhw_temp"]["first"] == 45.0
            assert data["training_data"]["dhw_temp"]["last"] == 50.0
            assert data["training_data"]["outdoor_temp"]["first"] == 5.0
            assert data["training_data"]["outdoor_temp"]["last"] == 10.0
            # Check hp_kwh_total includes delta
            assert data["training_data"]["hp_kwh_total"]["first"] == 1000.0
            assert data["training_data"]["hp_kwh_total"]["last"] == 1500.0
            assert data["training_data"]["hp_kwh_total"]["delta"] == 500.0
            # Check units are included (now extracted from source data)
            assert data["training_data"]["hp_kwh_total"]["unit"] == "kWh"
            assert data["training_data"]["dhw_temp"]["unit"] == "°C"

    def test_train_returns_empty_training_data_when_not_available(self, client):
        """Training returns empty training_data when no sensor data available."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = FeatureDatasetStats(
            total_slots=100,
            valid_slots=80,
            dropped_missing_features=10,
            dropped_missing_target=5,
            dropped_insufficient_history=5,
            features_used=["outdoor_temp", "wind"],
            has_7d_features=False,
            sensor_ranges={},  # Empty sensor ranges
            hp_kwh_delta=None,
            dhw_temp_range=None,
            hp_kwh_total_range=None,
        )
        mock_model = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.train_samples = 60
        mock_metrics.val_samples = 20
        mock_metrics.train_mae = 0.1
        mock_metrics.val_mae = 0.15
        mock_metrics.val_mape = 0.05
        mock_metrics.val_r2 = 0.85
        mock_metrics.features = ["outdoor_temp", "wind"]

        with patch("app.build_heating_feature_dataset") as mock_build, \
             patch("app.train_heating_demand_model") as mock_train:
            mock_build.return_value = (mock_df, mock_stats)
            mock_train.return_value = (mock_model, mock_metrics)

            response = client.post("/api/train/heating_demand")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "training_data" in data
            # training_data should be empty dict when no sensor ranges
            assert data["training_data"] == {}


class TestPredictHeatingDemandProfileEndpoint:
    """Test the /api/predictions/heating_demand_profile POST endpoint."""

    def test_predict_success(self, client, mock_model):
        """Successful prediction returns 200 with predictions."""
        scenario_features = [
            {"outdoor_temp": 5.0, "wind": 3.0, "humidity": 80.0},
            {"outdoor_temp": 6.0, "wind": 2.0, "humidity": 75.0},
        ]

        with patch("app._get_model") as mock_get_model, \
             patch("app.predict_scenario") as mock_predict:
            mock_get_model.return_value = mock_model
            mock_predict.return_value = [1.5, 1.2]

            response = client.post(
                "/api/predictions/heating_demand_profile",
                json={"scenario_features": scenario_features},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert len(data["predictions"]) == 2
            assert data["total_kwh"] == 2.7

    def test_predict_model_not_available(self, client):
        """Prediction without model returns 503."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = None

            response = client.post(
                "/api/predictions/heating_demand_profile",
                json={"scenario_features": [{"outdoor_temp": 5.0}]},
            )

            assert response.status_code == 503
            data = response.get_json()
            assert data["status"] == "error"
            assert "Model not trained" in data["message"]

    def test_predict_no_request_body(self, client, mock_model):
        """Prediction without valid JSON returns 400."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            # Send empty JSON object instead of no body
            response = client.post(
                "/api/predictions/heating_demand_profile",
                json={},
            )

            assert response.status_code == 400
            data = response.get_json()
            assert data["status"] == "error"

    def test_predict_empty_scenario(self, client, mock_model):
        """Prediction with empty scenario returns 400."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            response = client.post(
                "/api/predictions/heating_demand_profile",
                json={"scenario_features": []},
            )

            assert response.status_code == 400
            data = response.get_json()
            assert data["status"] == "error"
            assert "required" in data["message"] and "non-empty" in data["message"]

    def test_predict_missing_features(self, client, mock_model):
        """Prediction with missing features returns 400."""
        scenario_features = [
            {"outdoor_temp": 5.0},  # Missing wind and humidity
        ]

        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            response = client.post(
                "/api/predictions/heating_demand_profile",
                json={"scenario_features": scenario_features},
            )

            assert response.status_code == 400
            data = response.get_json()
            assert data["status"] == "error"
            assert "Missing" in data["message"]


class TestModelStatusEndpoint:
    """Test the /api/model/status GET endpoint."""

    def test_status_available(self, client, mock_model):
        """Model status returns available when model exists."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            response = client.get("/api/model/status")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "available"
            assert data["features"] == ["outdoor_temp", "wind", "humidity"]

    def test_status_not_available(self, client):
        """Model status returns not_available when no model."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = None

            response = client.get("/api/model/status")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "not_available"


class TestSensorInfoEndpoint:
    """Test the /api/sensors/info GET endpoint."""

    def test_sensors_info_success(self, client):
        """Get sensor info returns 200 with sensor list."""
        mock_sensors = [
            {
                "entity_id": "sensor.outdoor_temp",
                "first_timestamp": "2024-01-01T00:00:00",
                "last_timestamp": "2024-01-15T12:00:00",
                "sample_count": 1000,
            },
            {
                "entity_id": "sensor.wind",
                "first_timestamp": "2024-01-01T00:00:00",
                "last_timestamp": "2024-01-15T12:00:00",
                "sample_count": 500,
            },
        ]
        with patch("app.get_sensor_info") as mock_get_info:
            mock_get_info.return_value = mock_sensors

            response = client.get("/api/sensors/info")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["count"] == 2
            assert len(data["sensors"]) == 2
            assert data["sensors"][0]["entity_id"] == "sensor.outdoor_temp"

    def test_sensors_info_empty(self, client):
        """Get sensor info returns empty list when no sensors."""
        with patch("app.get_sensor_info") as mock_get_info:
            mock_get_info.return_value = []

            response = client.get("/api/sensors/info")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["count"] == 0
            assert data["sensors"] == []

    def test_sensors_info_error(self, client):
        """Get sensor info returns 500 on error."""
        with patch("app.get_sensor_info") as mock_get_info:
            mock_get_info.side_effect = Exception("Database error")

            response = client.get("/api/sensors/info")

            assert response.status_code == 500
            data = response.get_json()
            assert data["status"] == "error"


class TestEnrichScenarioEndpoint:
    """Test the /api/predictions/enrich_scenario POST endpoint."""

    def test_enrich_scenario_success(self, client):
        """Successful enrichment returns 200 with enriched features."""
        scenario = [
            {"outdoor_temp": 5.0, "indoor_temp": 20.0, "target_temp": 21.0},
            {"outdoor_temp": 4.5, "indoor_temp": 20.0, "target_temp": 21.0},
        ]

        response = client.post(
            "/api/predictions/enrich_scenario",
            json={"scenario_features": scenario},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert len(data["enriched_features"]) == 2
        assert "outdoor_temp_avg_1h" in data["enriched_features"][0]

    def test_enrich_scenario_with_timeslots(self, client):
        """Enrichment with timeslots adds time features."""
        scenario = [{"outdoor_temp": 5.0, "target_temp": 20.0}]
        timeslots = ["2024-01-15T14:00:00"]

        response = client.post(
            "/api/predictions/enrich_scenario",
            json={"scenario_features": scenario, "timeslots": timeslots},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["enriched_features"][0]["hour_of_day"] == 14

    def test_enrich_scenario_empty_features(self, client):
        """Enrichment with empty features returns 400."""
        response = client.post(
            "/api/predictions/enrich_scenario",
            json={"scenario_features": []},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"

    def test_enrich_scenario_no_body(self, client):
        """Enrichment without body returns 400."""
        response = client.post(
            "/api/predictions/enrich_scenario",
            json={},
        )

        assert response.status_code == 400


class TestCompareActualEndpoint:
    """Test the /api/predictions/compare_actual POST endpoint."""

    def test_compare_model_not_available(self, client):
        """Compare without model returns 503."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = None

            response = client.post(
                "/api/predictions/compare_actual",
                json={
                    "start_time": "2024-01-15T12:00:00",
                    "end_time": "2024-01-15T18:00:00",
                },
            )

            assert response.status_code == 503

    def test_compare_no_body(self, client, mock_model):
        """Compare without body returns 400."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            response = client.post(
                "/api/predictions/compare_actual",
                json={},
            )

            assert response.status_code == 400

    def test_compare_no_data(self, client, mock_model):
        """Compare with no data returns 404."""
        with patch("app._get_model") as mock_get_model, \
             patch("app.get_actual_vs_predicted_data") as mock_get_data:
            mock_get_model.return_value = mock_model
            mock_get_data.return_value = (None, "No data available")

            response = client.post(
                "/api/predictions/compare_actual",
                json={
                    "start_time": "2024-01-15T12:00:00",
                    "end_time": "2024-01-15T18:00:00",
                },
            )

            assert response.status_code == 404


class TestValidateStartTimeEndpoint:
    """Test the /api/predictions/validate_start_time POST endpoint."""

    def test_validate_valid_time(self, client):
        """Valid start time returns success."""
        from datetime import timedelta
        
        next_hour = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        with patch("app.validate_prediction_start_time") as mock_validate:
            mock_validate.return_value = (True, "Valid")

            response = client.post(
                "/api/predictions/validate_start_time",
                json={"start_time": next_hour.isoformat()},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["valid"] is True

    def test_validate_invalid_time(self, client):
        """Invalid start time returns valid=False."""
        with patch("app.validate_prediction_start_time") as mock_validate:
            mock_validate.return_value = (False, "Invalid")

            response = client.post(
                "/api/predictions/validate_start_time",
                json={"start_time": "2020-01-01T12:00:00"},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["valid"] is False

    def test_validate_no_body(self, client):
        """Validate without body returns 400."""
        response = client.post(
            "/api/predictions/validate_start_time",
            json={},
        )

        assert response.status_code == 400

    def test_validate_returns_next_valid_hour(self, client):
        """Response includes next valid hour."""
        from datetime import timedelta
        
        next_hour = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=2)

        with patch("app.validate_prediction_start_time") as mock_validate:
            mock_validate.return_value = (True, "Valid")

            response = client.post(
                "/api/predictions/validate_start_time",
                json={"start_time": next_hour.isoformat()},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert "next_valid_hour" in data


class TestTrainDatasetStatsTimeRange:
    """Test that training endpoint returns time range info in stats."""

    def test_train_returns_time_range_stats(self, client):
        """Training response includes data time range."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = FeatureDatasetStats(
            total_slots=100,
            valid_slots=80,
            dropped_missing_features=10,
            dropped_missing_target=5,
            dropped_insufficient_history=5,
            features_used=["outdoor_temp", "wind"],
            has_7d_features=True,
            data_start_time=datetime(2024, 1, 1, 0, 0, 0),
            data_end_time=datetime(2024, 1, 15, 0, 0, 0),
            available_history_hours=336.0,
        )
        mock_model = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.train_samples = 60
        mock_metrics.val_samples = 20
        mock_metrics.train_mae = 0.1
        mock_metrics.val_mae = 0.15
        mock_metrics.val_mape = 0.05
        mock_metrics.val_r2 = 0.85
        mock_metrics.features = ["outdoor_temp", "wind"]

        with patch("app.build_heating_feature_dataset") as mock_build, \
             patch("app.train_heating_demand_model") as mock_train:
            mock_build.return_value = (mock_df, mock_stats)
            mock_train.return_value = (mock_model, mock_metrics)

            response = client.post("/api/train/heating_demand")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "dataset_stats" in data
            assert data["dataset_stats"]["data_start_time"] == "2024-01-01T00:00:00"
            assert data["dataset_stats"]["data_end_time"] == "2024-01-15T00:00:00"
            assert data["dataset_stats"]["available_history_hours"] == 336.0


class TestScenarioPredictionEndpoint:
    """Test the /api/predictions/scenario POST endpoint."""

    def test_scenario_success(self, client, mock_model):
        """Successful scenario prediction returns 200 with predictions."""
        from datetime import timedelta
        
        next_hour = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        scenario_timeslots = [
            {
                "timestamp": (next_hour).isoformat(),
                "outdoor_temperature": 5.0,
                "wind_speed": 3.0,
                "humidity": 75.0,
                "pressure": 1013.0,
                "target_temperature": 20.0,
            },
            {
                "timestamp": (next_hour + timedelta(hours=1)).isoformat(),
                "outdoor_temperature": 4.5,
                "wind_speed": 3.5,
                "humidity": 76.0,
                "pressure": 1013.0,
                "target_temperature": 20.0,
            },
        ]

        with patch("app._get_model") as mock_get_model, \
             patch("app.predict_scenario") as mock_predict, \
             patch("app.convert_simplified_to_model_features") as mock_convert:
            mock_get_model.return_value = mock_model
            mock_predict.return_value = [1.5, 1.2]
            mock_convert.return_value = (
                [{"outdoor_temp": 5.0}, {"outdoor_temp": 4.5}],
                [next_hour, next_hour + timedelta(hours=1)],
            )

            response = client.post(
                "/api/predictions/scenario",
                json={"timeslots": scenario_timeslots},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert len(data["predictions"]) == 2
            assert data["total_kwh"] == 2.7
            assert data["slots_count"] == 2
            assert "timestamp" in data["predictions"][0]
            assert "predicted_kwh" in data["predictions"][0]

    def test_scenario_model_not_available(self, client):
        """Prediction without model returns 503."""
        from datetime import timedelta
        
        next_hour = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = None

            response = client.post(
                "/api/predictions/scenario",
                json={"timeslots": [{"timestamp": next_hour.isoformat()}]},
            )

            assert response.status_code == 503
            data = response.get_json()
            assert data["status"] == "error"
            assert "Model not trained" in data["message"]

    def test_scenario_no_request_body(self, client, mock_model):
        """Prediction without valid JSON returns 400."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            response = client.post(
                "/api/predictions/scenario",
                json={},
            )

            assert response.status_code == 400
            data = response.get_json()
            assert data["status"] == "error"
            assert "required" in data["message"]

    def test_scenario_empty_timeslots(self, client, mock_model):
        """Prediction with empty timeslots returns 400."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            response = client.post(
                "/api/predictions/scenario",
                json={"timeslots": []},
            )

            assert response.status_code == 400
            data = response.get_json()
            assert data["status"] == "error"
            assert "required" in data["message"] and "non-empty" in data["message"]

    def test_scenario_missing_required_fields(self, client, mock_model):
        """Prediction with missing required fields returns 400."""
        from datetime import timedelta
        
        next_hour = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            response = client.post(
                "/api/predictions/scenario",
                json={"timeslots": [
                    {"timestamp": next_hour.isoformat(), "outdoor_temperature": 5.0}
                    # Missing wind_speed, humidity, pressure, target_temperature
                ]},
            )

            assert response.status_code == 400
            data = response.get_json()
            assert data["status"] == "error"
            assert "errors" in data
            assert any("Missing required field" in err for err in data["errors"])

    def test_scenario_past_timestamp(self, client, mock_model):
        """Prediction with past timestamp returns 400."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            response = client.post(
                "/api/predictions/scenario",
                json={"timeslots": [
                    {
                        "timestamp": "2020-01-01T12:00:00",  # Past timestamp
                        "outdoor_temperature": 5.0,
                        "wind_speed": 3.0,
                        "humidity": 75.0,
                        "pressure": 1013.0,
                        "target_temperature": 20.0,
                    }
                ]},
            )

            assert response.status_code == 400
            data = response.get_json()
            assert data["status"] == "error"
            assert "errors" in data
            assert any("future" in err.lower() for err in data["errors"])

    def test_scenario_includes_required_fields_in_response(self, client, mock_model):
        """Error response includes required_fields."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            response = client.post(
                "/api/predictions/scenario",
                json={"timeslots": []},
            )

            assert response.status_code == 400
            data = response.get_json()
            assert "required_fields" in data
            assert "timestamp" in data["required_fields"]
            assert "outdoor_temperature" in data["required_fields"]


class TestScenarioExampleEndpoint:
    """Test the /api/examples/scenario GET endpoint."""

    def test_scenario_example_success(self, client):
        """Get scenario example returns 200 with 24-hour example."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = None

            response = client.get("/api/examples/scenario")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "example" in data
            assert "timeslots" in data["example"]
            assert len(data["example"]["timeslots"]) == 24
            assert data["model_available"] is False
            assert "required_fields" in data
            assert "optional_fields" in data

    def test_scenario_example_structure(self, client):
        """Scenario example has correct structure for each timeslot."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = None

            response = client.get("/api/examples/scenario")

            data = response.get_json()
            slot = data["example"]["timeslots"][0]
            
            # Check all required fields are present
            assert "timestamp" in slot
            assert "outdoor_temperature" in slot
            assert "wind_speed" in slot
            assert "humidity" in slot
            assert "pressure" in slot
            assert "target_temperature" in slot

    def test_scenario_example_with_model(self, client, mock_model):
        """Scenario example shows model_available when model exists."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            response = client.get("/api/examples/scenario")

            data = response.get_json()
            assert data["model_available"] is True

    def test_scenario_example_timestamps_are_future(self, client):
        """Scenario example timestamps are in the future."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = None

            response = client.get("/api/examples/scenario")

            data = response.get_json()
            now = datetime.now()
            
            for slot in data["example"]["timeslots"]:
                ts = datetime.fromisoformat(slot["timestamp"])
                assert ts > now


class TestAvailableDaysEndpoint:
    """Test the /api/examples/available_days GET endpoint."""

    def test_available_days_success(self, client):
        """Get available days returns 200 with list of days."""
        with patch("app.get_available_historical_days") as mock_get_days:
            mock_get_days.return_value = ["2024-01-02", "2024-01-03", "2024-01-04"]

            response = client.get("/api/examples/available_days")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "days" in data
            assert len(data["days"]) == 3
            assert data["count"] == 3
            assert "2024-01-02" in data["days"]

    def test_available_days_empty(self, client):
        """Get available days returns empty list when no data."""
        with patch("app.get_available_historical_days") as mock_get_days:
            mock_get_days.return_value = []

            response = client.get("/api/examples/available_days")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["days"] == []
            assert data["count"] == 0

    def test_available_days_error(self, client):
        """Error fetching days returns 500."""
        with patch("app.get_available_historical_days") as mock_get_days:
            mock_get_days.side_effect = Exception("Database error")

            response = client.get("/api/examples/available_days")

            assert response.status_code == 500
            data = response.get_json()
            assert data["status"] == "error"
            assert "Database error" in data["message"]


class TestHistoricalDayEndpoint:
    """Test the /api/examples/historical_day/<date_str> GET endpoint."""

    def test_historical_day_success(self, client):
        """Get historical day returns 200 with hourly data."""
        hourly_data = [
            {
                "timestamp": "2024-01-15T00:00:00",
                "outdoor_temperature": 5.0,
                "wind_speed": 3.0,
                "humidity": 75.0,
                "pressure": 1013.0,
                "target_temperature": 17.0,
                "indoor_temperature": 19.5,
                "actual_heating_kwh": 1.25,
            },
            {
                "timestamp": "2024-01-15T01:00:00",
                "outdoor_temperature": 4.5,
                "wind_speed": 3.5,
                "humidity": 76.0,
                "pressure": 1013.0,
                "target_temperature": 17.0,
                "indoor_temperature": 19.3,
                "actual_heating_kwh": 1.30,
            },
        ]
        with patch("app.get_historical_day_hourly_data") as mock_get_data, \
             patch("app._get_model") as mock_get_model:
            mock_get_data.return_value = (hourly_data, None)
            mock_get_model.return_value = None

            response = client.get("/api/examples/historical_day/2024-01-15")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["date"] == "2024-01-15"
            assert "hourly_data" in data
            assert len(data["hourly_data"]) == 2
            assert "scenario_format" in data
            assert "timeslots" in data["scenario_format"]

    def test_historical_day_not_found(self, client):
        """Get historical day returns 404 when no data available."""
        with patch("app.get_historical_day_hourly_data") as mock_get_data:
            mock_get_data.return_value = (None, "No data available for date 2024-01-15")

            response = client.get("/api/examples/historical_day/2024-01-15")

            assert response.status_code == 404
            data = response.get_json()
            assert data["status"] == "error"
            assert "No data" in data["message"]

    def test_historical_day_invalid_date(self, client):
        """Get historical day with invalid date format returns 404."""
        with patch("app.get_historical_day_hourly_data") as mock_get_data:
            mock_get_data.return_value = (None, "Invalid date format")

            response = client.get("/api/examples/historical_day/invalid-date")

            assert response.status_code == 404
            data = response.get_json()
            assert data["status"] == "error"

    def test_historical_day_scenario_format(self, client):
        """Historical day returns proper scenario format."""
        hourly_data = [
            {
                "timestamp": "2024-01-15T12:00:00",
                "outdoor_temperature": 8.0,
                "wind_speed": 4.0,
                "humidity": 70.0,
                "pressure": 1015.0,
                "target_temperature": 20.0,
                "indoor_temperature": 19.8,
                "actual_heating_kwh": 0.95,
            },
        ]
        with patch("app.get_historical_day_hourly_data") as mock_get_data, \
             patch("app._get_model") as mock_get_model:
            mock_get_data.return_value = (hourly_data, None)
            mock_get_model.return_value = None

            response = client.get("/api/examples/historical_day/2024-01-15")

            data = response.get_json()
            scenario_slot = data["scenario_format"]["timeslots"][0]
            
            # Check that scenario format contains required fields
            assert "timestamp" in scenario_slot
            assert "outdoor_temperature" in scenario_slot
            assert "wind_speed" in scenario_slot
            assert "humidity" in scenario_slot
            assert "pressure" in scenario_slot
            assert "target_temperature" in scenario_slot

    def test_historical_day_model_status(self, client, mock_model):
        """Historical day shows model availability."""
        hourly_data = [{"timestamp": "2024-01-15T00:00:00", "outdoor_temperature": 5.0}]
        
        with patch("app.get_historical_day_hourly_data") as mock_get_data, \
             patch("app._get_model") as mock_get_model:
            mock_get_data.return_value = (hourly_data, None)
            mock_get_model.return_value = mock_model

            response = client.get("/api/examples/historical_day/2024-01-15")

            data = response.get_json()
            assert data["model_available"] is True

    def test_historical_day_error(self, client):
        """Error fetching historical day returns 500."""
        with patch("app.get_historical_day_hourly_data") as mock_get_data:
            mock_get_data.side_effect = Exception("Database error")

            response = client.get("/api/examples/historical_day/2024-01-15")

            assert response.status_code == 500
            data = response.get_json()
            assert data["status"] == "error"
            assert "Database error" in data["message"]

    def test_historical_day_scenario_timestamp_adjusted(self, client):
        """Scenario format timestamps are adjusted to 2 days after today."""
        from datetime import timedelta
        
        hourly_data = [
            {
                "timestamp": "2024-01-15T12:00:00",
                "outdoor_temperature": 8.0,
            },
            {
                "timestamp": "2024-01-15T13:00:00",
                "outdoor_temperature": 9.0,
            },
        ]
        with patch("app.get_historical_day_hourly_data") as mock_get_data, \
             patch("app._get_model") as mock_get_model:
            mock_get_data.return_value = (hourly_data, None)
            mock_get_model.return_value = None

            response = client.get("/api/examples/historical_day/2024-01-15")

            data = response.get_json()
            
            # Calculate expected prediction date (2 days from today)
            today = datetime.now().date()
            prediction_date = today + timedelta(days=2)
            
            # Check that scenario_format timestamps are adjusted
            scenario_ts_1 = datetime.fromisoformat(
                data["scenario_format"]["timeslots"][0]["timestamp"]
            )
            scenario_ts_2 = datetime.fromisoformat(
                data["scenario_format"]["timeslots"][1]["timestamp"]
            )
            
            # Verify the date is 2 days after today
            assert scenario_ts_1.date() == prediction_date
            assert scenario_ts_2.date() == prediction_date
            
            # Verify the hours are preserved
            assert scenario_ts_1.hour == 12
            assert scenario_ts_2.hour == 13
            
            # Verify original hourly_data timestamps remain unchanged
            assert data["hourly_data"][0]["timestamp"] == "2024-01-15T12:00:00"
            assert data["hourly_data"][1]["timestamp"] == "2024-01-15T13:00:00"


class TestSampleRateEndpoint:
    """Test the /api/sample_rate GET endpoint."""

    def test_get_sample_rate(self, client):
        """Get sample rate returns current configuration."""
        with patch("app.get_sample_rate_minutes") as mock_get_rate:
            mock_get_rate.return_value = 5

            response = client.get("/api/sample_rate")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["sample_rate_minutes"] == 5
            assert "5-minute" in data["description"]

    def test_get_sample_rate_custom(self, client):
        """Get sample rate returns custom configuration."""
        with patch("app.get_sample_rate_minutes") as mock_get_rate:
            mock_get_rate.return_value = 15

            response = client.get("/api/sample_rate")

            assert response.status_code == 200
            data = response.get_json()
            assert data["sample_rate_minutes"] == 15
            assert "15-minute" in data["description"]


class TestResampleWithSampleRate:
    """Test the /resample POST endpoint with configurable sample rate."""

    def test_resample_with_custom_rate(self, client):
        """Resample with custom sample rate accepts the value and starts background thread."""
        response = client.post(
            "/resample",
            json={"sample_rate_minutes": 10},
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["running"] is True
        assert "background" in data["message"].lower()

    def test_resample_invalid_rate_not_divisor(self, client):
        """Resample with sample rate that doesn't divide 60 returns error."""
        response = client.post(
            "/resample",
            json={"sample_rate_minutes": 7},
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
        # Check that the error message mentions valid rates
        assert "[1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]" in data["message"]

    def test_resample_invalid_rate_too_high(self, client):
        """Resample with sample rate > 60 returns error."""
        response = client.post(
            "/resample",
            json={"sample_rate_minutes": 120},
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"

    def test_resample_default_rate(self, client):
        """Resample without custom rate uses default and starts background thread."""
        response = client.post("/resample")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["running"] is True

    def test_resample_with_flush(self, client):
        """Resample with flush=true accepts the parameter and starts background thread."""
        response = client.post(
            "/resample",
            json={"sample_rate_minutes": 10, "flush": True},
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["running"] is True

    def test_resample_table_flushed_in_response(self, client):
        """Resample with flush starts background thread."""
        response = client.post(
            "/resample",
            json={"sample_rate_minutes": 5, "flush": True},
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["running"] is True

    def test_resample_with_flush_default_rate(self, client):
        """Resample with flush and default rate starts background thread."""
        response = client.post(
            "/resample",
            json={"flush": True},
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["running"] is True


class TestResampleStatusEndpoint:
    """Test the /api/resample/status GET endpoint."""

    def test_resample_status_no_progress(self, client):
        """Status endpoint returns success when no resampling has been run."""
        response = client.get("/api/resample/status")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["running"] is False
        assert data["progress"] is None
        assert "message" in data

    def test_resample_status_with_progress(self, client):
        """Status endpoint returns progress when resampling is running."""
        # Simulate resampling in progress by setting global state
        with patch("app._resample_progress") as mock_progress, \
             patch("app._resample_running", True), \
             patch("app._resample_lock"):
            
            from db.resample import ResampleProgress
            from datetime import datetime
            
            mock_progress.phase = "resampling"
            mock_progress.slots_processed = 60
            mock_progress.slots_total = 120
            mock_progress.slots_saved = 55
            mock_progress.slots_skipped = 5
            mock_progress.categories = ["outdoor_temp", "indoor_temp"]
            mock_progress.current_slot = datetime(2024, 12, 3, 10, 0, 0)
            mock_progress.log_messages = ["Log line 1", "Log line 2"]
            mock_progress.sample_rate_minutes = 5
            mock_progress.get_hours_processed = lambda: 5
            mock_progress.get_hours_total = lambda: 10
            mock_progress.error_message = None
            
            response = client.get("/api/resample/status")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["running"] is True
            assert data["progress"]["phase"] == "resampling"
            assert data["progress"]["hours_processed"] == 5
            assert data["progress"]["hours_total"] == 10
            assert data["progress"]["slots_processed"] == 60
            assert data["progress"]["slots_total"] == 120


class TestSampleRateEndpoints:
    """Test the /api/sample_rate GET and POST endpoints."""

    def test_get_sample_rate_success(self, client):
        """Get sample rate returns 200 with current rate."""
        with patch("app.get_sample_rate_minutes") as mock_get_rate:
            mock_get_rate.return_value = 5

            response = client.get("/api/sample_rate")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["sample_rate_minutes"] == 5
            assert "valid_rates" in data
            assert 5 in data["valid_rates"]

    def test_get_sample_rate_custom_value(self, client):
        """Get sample rate returns custom configured rate."""
        with patch("app.get_sample_rate_minutes") as mock_get_rate:
            mock_get_rate.return_value = 15

            response = client.get("/api/sample_rate")

            assert response.status_code == 200
            data = response.get_json()
            assert data["sample_rate_minutes"] == 15
            assert "15-minute" in data["description"]

    def test_get_sample_rate_error(self, client):
        """Get sample rate returns 500 on error."""
        with patch("app.get_sample_rate_minutes") as mock_get_rate:
            mock_get_rate.side_effect = Exception("Config error")

            response = client.get("/api/sample_rate")

            assert response.status_code == 500
            data = response.get_json()
            assert data["status"] == "error"
            assert "Config error" in data["message"]

    def test_update_sample_rate_success(self, client):
        """Update sample rate returns 200 with new rate."""
        with patch("app.set_sample_rate_minutes") as mock_set_rate:
            mock_set_rate.return_value = True

            response = client.post(
                "/api/sample_rate",
                json={"sample_rate_minutes": 10},
                content_type="application/json",
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["sample_rate_minutes"] == 10
            assert "10 minutes" in data["message"]
            assert "note" in data
            mock_set_rate.assert_called_once_with(10)

    def test_update_sample_rate_no_body(self, client):
        """Update sample rate without body returns 400."""
        response = client.post("/api/sample_rate")

        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
        assert "valid_rates" in data

    def test_update_sample_rate_missing_field(self, client):
        """Update sample rate without sample_rate_minutes returns 400."""
        response = client.post(
            "/api/sample_rate",
            json={"other_field": 10},
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
        assert "sample_rate_minutes is required" in data["message"]

    def test_update_sample_rate_invalid_value(self, client):
        """Update sample rate with invalid value returns 400."""
        response = client.post(
            "/api/sample_rate",
            json={"sample_rate_minutes": 7},  # 7 is not a valid divisor of 60
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
        assert "must be one of" in data["message"]
        assert "valid_rates" in data

    def test_update_sample_rate_invalid_type(self, client):
        """Update sample rate with non-integer returns 400."""
        response = client.post(
            "/api/sample_rate",
            json={"sample_rate_minutes": "five"},
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
        assert "must be an integer" in data["message"]

    def test_update_sample_rate_save_failure(self, client):
        """Update sample rate returns 500 if save fails."""
        with patch("app.set_sample_rate_minutes") as mock_set_rate:
            mock_set_rate.return_value = False

            response = client.post(
                "/api/sample_rate",
                json={"sample_rate_minutes": 10},
                content_type="application/json",
            )

            assert response.status_code == 500
            data = response.get_json()
            assert data["status"] == "error"
            assert "Failed to save" in data["message"]

    def test_update_sample_rate_all_valid_rates(self, client):
        """Update sample rate works with all valid rates."""
        valid_rates = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
        
        for rate in valid_rates:
            with patch("app.set_sample_rate_minutes") as mock_set_rate:
                mock_set_rate.return_value = True

                response = client.post(
                    "/api/sample_rate",
                    json={"sample_rate_minutes": rate},
                    content_type="application/json",
                )

                assert response.status_code == 200, f"Failed for rate {rate}"
                data = response.get_json()
                assert data["sample_rate_minutes"] == rate


class TestScenarioPredictionWithTwoStep:
    """Test the /api/predictions/scenario POST endpoint with two-step prediction."""

    @pytest.fixture
    def mock_two_step_model(self):
        """Create a mock two-step heating demand model."""
        model = MagicMock()
        model.is_available = True
        model.feature_names = ["outdoor_temp", "wind", "humidity"]
        model.training_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        model.activity_threshold_kwh = 0.05
        return model

    @pytest.fixture
    def mock_feature_config_two_step_enabled(self):
        """Create a mock feature config with two-step prediction enabled."""
        config = MagicMock()
        config.is_two_step_prediction_enabled.return_value = True
        return config

    @pytest.fixture
    def mock_feature_config_two_step_disabled(self):
        """Create a mock feature config with two-step prediction disabled."""
        config = MagicMock()
        config.is_two_step_prediction_enabled.return_value = False
        return config

    def test_scenario_uses_two_step_when_enabled(
        self, client, mock_two_step_model, mock_feature_config_two_step_enabled
    ):
        """Scenario prediction uses two-step model when enabled and available."""
        from datetime import timedelta
        from ml.two_step_model import TwoStepPrediction
        
        next_hour = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        scenario_timeslots = [
            {
                "timestamp": next_hour.isoformat(),
                "outdoor_temperature": 5.0,
                "wind_speed": 3.0,
                "humidity": 75.0,
                "pressure": 1013.0,
                "target_temperature": 20.0,
            },
            {
                "timestamp": (next_hour + timedelta(hours=1)).isoformat(),
                "outdoor_temperature": 4.5,
                "wind_speed": 3.5,
                "humidity": 76.0,
                "pressure": 1013.0,
                "target_temperature": 20.0,
            },
        ]

        mock_predictions = [
            TwoStepPrediction(is_active=True, predicted_kwh=1.5, classifier_probability=0.85),
            TwoStepPrediction(is_active=False, predicted_kwh=0.0, classifier_probability=0.2),
        ]

        with patch("app.get_feature_config") as mock_get_config, \
             patch("app._get_two_step_model") as mock_get_two_step, \
             patch("app.predict_two_step_scenario") as mock_predict, \
             patch("app.convert_simplified_to_model_features") as mock_convert:
            mock_get_config.return_value = mock_feature_config_two_step_enabled
            mock_get_two_step.return_value = mock_two_step_model
            mock_predict.return_value = mock_predictions
            mock_convert.return_value = (
                [{"outdoor_temp": 5.0}, {"outdoor_temp": 4.5}],
                [next_hour, next_hour + timedelta(hours=1)],
            )

            response = client.post(
                "/api/predictions/scenario",
                json={"timeslots": scenario_timeslots},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert len(data["predictions"]) == 2
            assert data["total_kwh"] == 1.5
            assert data["slots_count"] == 2
            
            # Check two-step specific fields
            assert data["model_info"]["two_step_prediction"] is True
            assert data["model_info"]["activity_threshold_kwh"] == 0.05
            assert "summary" in data
            assert data["summary"]["active_hours"] == 1
            assert data["summary"]["inactive_hours"] == 1
            
            # Check prediction details
            assert data["predictions"][0]["is_active"] is True
            assert data["predictions"][0]["activity_probability"] == 0.85
            assert data["predictions"][1]["is_active"] is False
            assert data["predictions"][1]["predicted_kwh"] == 0.0

    def test_scenario_uses_single_step_when_disabled(
        self, client, mock_model, mock_feature_config_two_step_disabled
    ):
        """Scenario prediction uses single-step model when two-step is disabled."""
        from datetime import timedelta
        
        next_hour = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        scenario_timeslots = [
            {
                "timestamp": next_hour.isoformat(),
                "outdoor_temperature": 5.0,
                "wind_speed": 3.0,
                "humidity": 75.0,
                "pressure": 1013.0,
                "target_temperature": 20.0,
            },
        ]

        with patch("app.get_feature_config") as mock_get_config, \
             patch("app._get_model") as mock_get_model, \
             patch("app.predict_scenario") as mock_predict, \
             patch("app.convert_simplified_to_model_features") as mock_convert:
            mock_get_config.return_value = mock_feature_config_two_step_disabled
            mock_get_model.return_value = mock_model
            mock_predict.return_value = [1.5]
            mock_convert.return_value = (
                [{"outdoor_temp": 5.0}],
                [next_hour],
            )

            response = client.post(
                "/api/predictions/scenario",
                json={"timeslots": scenario_timeslots},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["model_info"]["two_step_prediction"] is False
            
            # Check that two-step specific fields are NOT present
            assert "is_active" not in data["predictions"][0]
            assert "summary" not in data

    def test_scenario_falls_back_when_two_step_model_not_available(
        self, client, mock_model, mock_feature_config_two_step_enabled
    ):
        """Falls back to single-step when two-step is enabled but model not available."""
        from datetime import timedelta
        
        next_hour = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        scenario_timeslots = [
            {
                "timestamp": next_hour.isoformat(),
                "outdoor_temperature": 5.0,
                "wind_speed": 3.0,
                "humidity": 75.0,
                "pressure": 1013.0,
                "target_temperature": 20.0,
            },
        ]

        with patch("app.get_feature_config") as mock_get_config, \
             patch("app._get_two_step_model") as mock_get_two_step, \
             patch("app._get_model") as mock_get_model, \
             patch("app.predict_scenario") as mock_predict, \
             patch("app.convert_simplified_to_model_features") as mock_convert:
            mock_get_config.return_value = mock_feature_config_two_step_enabled
            mock_get_two_step.return_value = None  # Two-step model not available
            mock_get_model.return_value = mock_model
            mock_predict.return_value = [1.5]
            mock_convert.return_value = (
                [{"outdoor_temp": 5.0}],
                [next_hour],
            )

            response = client.post(
                "/api/predictions/scenario",
                json={"timeslots": scenario_timeslots},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["model_info"]["two_step_prediction"] is False

    def test_scenario_returns_503_when_no_models_available(
        self, client, mock_feature_config_two_step_enabled
    ):
        """Returns 503 when two-step is enabled but no models are available."""
        from datetime import timedelta
        
        next_hour = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        scenario_timeslots = [
            {
                "timestamp": next_hour.isoformat(),
                "outdoor_temperature": 5.0,
                "wind_speed": 3.0,
                "humidity": 75.0,
                "pressure": 1013.0,
                "target_temperature": 20.0,
            },
        ]

        with patch("app.get_feature_config") as mock_get_config, \
             patch("app._get_two_step_model") as mock_get_two_step, \
             patch("app._get_model") as mock_get_model:
            mock_get_config.return_value = mock_feature_config_two_step_enabled
            mock_get_two_step.return_value = None
            mock_get_model.return_value = None

            response = client.post(
                "/api/predictions/scenario",
                json={"timeslots": scenario_timeslots},
            )

            assert response.status_code == 503
            data = response.get_json()
            assert data["status"] == "error"
            assert "Model not trained" in data["message"]


class TestTrainTwoStepHeatingDemandEndpoint:
    """Test the /api/train/two_step_heating_demand POST endpoint."""

    def test_train_two_step_success(self, client):
        """Successful two-step training returns 200 with classifier and regressor metrics."""
        mock_df = pd.DataFrame({
            "outdoor_temp": [10.0, 11.0],
            "target_heating_kwh_1h": [1.0, 1.5],
        })
        mock_stats = FeatureDatasetStats(
            total_slots=100,
            valid_slots=80,
            dropped_missing_features=10,
            dropped_missing_target=5,
            dropped_insufficient_history=5,
            features_used=["outdoor_temp", "wind"],
            has_7d_features=False,
        )
        mock_model = MagicMock()
        mock_model.is_available = True
        mock_metrics = MagicMock()
        mock_metrics.computed_threshold_kwh = 0.05
        mock_metrics.active_samples = 250
        mock_metrics.inactive_samples = 50
        mock_metrics.classifier_accuracy = 0.92
        mock_metrics.classifier_precision = 0.88
        mock_metrics.classifier_recall = 0.95
        mock_metrics.classifier_f1 = 0.91
        mock_metrics.classifier_description = "Predicts active/inactive"
        mock_metrics.regressor_train_samples = 200
        mock_metrics.regressor_val_samples = 50
        mock_metrics.regressor_train_mae = 0.15
        mock_metrics.regressor_val_mae = 0.18
        mock_metrics.regressor_val_mape = 0.125
        mock_metrics.regressor_val_r2 = 0.85
        mock_metrics.regressor_description = "Predicts kWh for active hours"
        mock_metrics.features = ["outdoor_temp", "wind"]

        with patch("app.build_heating_feature_dataset") as mock_build, \
             patch("app.train_two_step_heating_demand_model") as mock_train:
            mock_build.return_value = (mock_df, mock_stats)
            mock_train.return_value = (mock_model, mock_metrics)

            response = client.post("/api/train/two_step_heating_demand")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            
            # Verify threshold info is present
            assert "threshold" in data
            assert data["threshold"]["computed_threshold_kwh"] == 0.05
            assert data["threshold"]["active_samples"] == 250
            assert data["threshold"]["inactive_samples"] == 50
            
            # Verify top-level classifier_metrics for UI compatibility
            assert "classifier_metrics" in data
            assert data["classifier_metrics"]["accuracy"] == 0.92
            assert data["classifier_metrics"]["precision"] == 0.88
            assert data["classifier_metrics"]["recall"] == 0.95
            assert data["classifier_metrics"]["f1"] == 0.91
            
            # Verify top-level regressor_metrics for UI compatibility
            assert "regressor_metrics" in data
            assert data["regressor_metrics"]["train_samples"] == 200
            assert data["regressor_metrics"]["val_samples"] == 50
            assert data["regressor_metrics"]["train_mae_kwh"] == 0.15
            assert data["regressor_metrics"]["val_mae_kwh"] == 0.18
            assert data["regressor_metrics"]["val_mape_pct"] == 12.5
            assert data["regressor_metrics"]["val_r2"] == 0.85
            
            # Verify detailed step1_classifier info
            assert "step1_classifier" in data
            assert data["step1_classifier"]["description"] == "Predicts active/inactive"
            assert data["step1_classifier"]["metrics"]["accuracy"] == 0.92
            
            # Verify detailed step2_regressor info
            assert "step2_regressor" in data
            assert data["step2_regressor"]["description"] == "Predicts kWh for active hours"
            assert data["step2_regressor"]["metrics"]["train_mae_kwh"] == 0.15

    def test_train_two_step_insufficient_data(self, client):
        """Two-step training with insufficient data returns 400."""
        mock_stats = FeatureDatasetStats(
            total_slots=30,
            valid_slots=10,
            dropped_missing_features=10,
            dropped_missing_target=5,
            dropped_insufficient_history=5,
            features_used=[],
            has_7d_features=False,
        )

        with patch("app.build_heating_feature_dataset") as mock_build:
            mock_build.return_value = (None, mock_stats)

            response = client.post("/api/train/two_step_heating_demand")

            assert response.status_code == 400
            data = response.get_json()
            assert data["status"] == "error"
            assert "Insufficient data" in data["message"]


# =============================================================================
# SENSOR CONFIGURATION ENDPOINTS TESTS
# =============================================================================


class TestSensorCategoryConfigEndpoint:
    """Test the /api/sensors/category_config GET endpoint."""

    def test_get_config_success(self, client, tmp_path):
        """Should return sensor configuration."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                response = client.get("/api/sensors/category_config")

                assert response.status_code == 200
                data = response.get_json()
                assert data["status"] == "success"
                assert "config" in data
                assert "sensors_by_type" in data
                assert "enabled_entity_ids" in data
                assert data["config"]["core_sensor_count"] > 0
                assert data["config"]["experimental_sensor_count"] > 0

    def test_config_has_sensor_types(self, client, tmp_path):
        """Should have sensors grouped by type."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                response = client.get("/api/sensors/category_config")

                assert response.status_code == 200
                data = response.get_json()
                sensors_by_type = data["sensors_by_type"]
                
                # Should have at least weather and usage types
                assert "weather" in sensors_by_type
                assert "usage" in sensors_by_type


class TestSensorToggleEndpoint:
    """Test the /api/sensors/toggle POST endpoint."""

    def test_toggle_experimental_sensor_enable(self, client, tmp_path):
        """Should enable an experimental sensor."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                response = client.post(
                    "/api/sensors/toggle",
                    json={"category_name": "pressure", "enabled": True},
                )

                assert response.status_code == 200
                data = response.get_json()
                assert data["status"] == "success"
                assert "enabled" in data["message"]
                assert "pressure" in data["enabled_sensors"]

    def test_toggle_experimental_sensor_disable(self, client, tmp_path):
        """Should disable an experimental sensor."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                # First enable
                client.post(
                    "/api/sensors/toggle",
                    json={"category_name": "pressure", "enabled": True},
                )
                
                # Then disable
                response = client.post(
                    "/api/sensors/toggle",
                    json={"category_name": "pressure", "enabled": False},
                )

                assert response.status_code == 200
                data = response.get_json()
                assert data["status"] == "success"
                assert "disabled" in data["message"]
                assert "pressure" not in data["enabled_sensors"]

    def test_cannot_toggle_core_sensor(self, client, tmp_path):
        """Should not be able to toggle a core sensor."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                response = client.post(
                    "/api/sensors/toggle",
                    json={"category_name": "outdoor_temp", "enabled": False},
                )

                assert response.status_code == 400
                data = response.get_json()
                assert data["status"] == "error"
                assert "core sensor" in data["message"].lower()

    def test_toggle_unknown_sensor(self, client, tmp_path):
        """Should return error for unknown sensor."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                response = client.post(
                    "/api/sensors/toggle",
                    json={"category_name": "nonexistent", "enabled": True},
                )

                assert response.status_code == 400
                data = response.get_json()
                assert data["status"] == "error"
                assert "Unknown" in data["message"]

    def test_toggle_missing_category_name(self, client):
        """Should return error when category_name is missing."""
        response = client.post(
            "/api/sensors/toggle",
            json={"enabled": True},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
        assert "category_name" in data["message"]

    def test_toggle_missing_enabled(self, client):
        """Should return error when enabled is missing."""
        response = client.post(
            "/api/sensors/toggle",
            json={"category_name": "pressure"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
        assert "enabled" in data["message"]


class TestSensorSetEntityEndpoint:
    """Test the /api/sensors/set_entity POST endpoint."""

    def test_set_entity_success(self, client, tmp_path):
        """Should set entity ID for a sensor."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                response = client.post(
                    "/api/sensors/set_entity",
                    json={
                        "category_name": "outdoor_temp",
                        "entity_id": "sensor.custom_temperature",
                    },
                )

                assert response.status_code == 200
                data = response.get_json()
                assert data["status"] == "success"
                assert "sensor.custom_temperature" in data["message"]
                assert data["sensor"]["entity_id"] == "sensor.custom_temperature"

    def test_set_entity_for_experimental_sensor(self, client, tmp_path):
        """Should set entity ID for experimental sensor."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                response = client.post(
                    "/api/sensors/set_entity",
                    json={
                        "category_name": "pressure",
                        "entity_id": "sensor.my_pressure",
                    },
                )

                assert response.status_code == 200
                data = response.get_json()
                assert data["status"] == "success"
                assert data["sensor"]["is_core"] is False

    def test_set_entity_unknown_sensor(self, client, tmp_path):
        """Should return error for unknown sensor."""
        config_file = tmp_path / "sensor_category_config.json"
        
        with patch("db.sensor_category_config.SENSOR_CONFIG_FILE_PATH", config_file):
            with patch("db.sensor_category_config._config", None):
                response = client.post(
                    "/api/sensors/set_entity",
                    json={
                        "category_name": "nonexistent",
                        "entity_id": "sensor.test",
                    },
                )

                assert response.status_code == 400
                data = response.get_json()
                assert data["status"] == "error"
                assert "Unknown" in data["message"]

    def test_set_entity_empty_entity_id(self, client):
        """Should return error for empty entity_id."""
        response = client.post(
            "/api/sensors/set_entity",
            json={
                "category_name": "outdoor_temp",
                "entity_id": "",
            },
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
        assert "entity_id" in data["message"]

    def test_set_entity_missing_category_name(self, client):
        """Should return error when category_name is missing."""
        response = client.post(
            "/api/sensors/set_entity",
            json={"entity_id": "sensor.test"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "error"
        assert "category_name" in data["message"]


class TestSensorDefinitionsEndpoint:
    """Test the /api/sensors/definitions GET endpoint."""

    def test_get_definitions_success(self, client):
        """Should return sensor definitions."""
        response = client.get("/api/sensors/definitions")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert "core_sensors" in data
        assert "experimental_sensors" in data
        assert "total_count" in data
        assert len(data["core_sensors"]) > 0
        assert len(data["experimental_sensors"]) > 0

    def test_definitions_have_required_fields(self, client):
        """Sensor definitions should have required fields."""
        response = client.get("/api/sensors/definitions")

        assert response.status_code == 200
        data = response.get_json()
        
        # Check a core sensor
        core_sensor = data["core_sensors"][0]
        assert "category_name" in core_sensor
        assert "display_name" in core_sensor
        assert "description" in core_sensor
        assert "unit" in core_sensor
        assert "is_core" in core_sensor
        assert core_sensor["is_core"] is True

        # Check an experimental sensor
        exp_sensor = data["experimental_sensors"][0]
        assert exp_sensor["is_core"] is False

    def test_hp_kwh_total_in_core(self, client):
        """hp_kwh_total should be in core sensors."""
        response = client.get("/api/sensors/definitions")

        assert response.status_code == 200
        data = response.get_json()
        
        core_names = [s["category_name"] for s in data["core_sensors"]]
        assert "hp_kwh_total" in core_names
