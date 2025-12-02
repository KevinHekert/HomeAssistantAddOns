"""
Tests for Flask app endpoints.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd

from app import app
from db.resample import ResampleStats
from ml.heating_features import FeatureDatasetStats


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
    """Test the /resample POST endpoint."""

    def test_resample_success(self, client):
        """Successful resample returns 200 with success message and stats."""
        mock_stats = ResampleStats(
            slots_processed=100,
            slots_saved=90,
            slots_skipped=10,
            categories=["outdoor_temp", "wind"],
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 20, 0, 0),
        )
        with patch("app.resample_all_categories") as mock_resample:
            mock_resample.return_value = mock_stats

            response = client.post("/resample")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "successfully" in data["message"]
            assert data["stats"]["slots_processed"] == 100
            assert data["stats"]["slots_saved"] == 90
            assert data["stats"]["slots_skipped"] == 10
            assert data["stats"]["categories"] == ["outdoor_temp", "wind"]
            mock_resample.assert_called_once()

    def test_resample_error(self, client):
        """Error during resample returns 500 with error message."""
        with patch("app.resample_all_categories") as mock_resample:
            mock_resample.side_effect = Exception("Database error")

            response = client.post("/resample")

            assert response.status_code == 500
            data = response.get_json()
            assert data["status"] == "error"
            assert "Database error" in data["message"]


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


class TestSingleSlotExampleEndpoint:
    """Test the /api/examples/single_slot GET endpoint."""

    def test_single_slot_example_no_model(self, client):
        """Get single slot example without model returns basic example."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = None

            response = client.get("/api/examples/single_slot")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "example" in data
            assert "scenario_features" in data["example"]
            assert len(data["example"]["scenario_features"]) == 1
            assert data["model_available"] is False

    def test_single_slot_example_with_model(self, client, mock_model):
        """Get single slot example with model returns example with all features."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            response = client.get("/api/examples/single_slot")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["model_available"] is True
            assert data["required_features"] == ["outdoor_temp", "wind", "humidity"]
            # Check that example contains the model features
            features = data["example"]["scenario_features"][0]
            assert "outdoor_temp" in features
            assert "wind" in features
            assert "humidity" in features


class TestFullDayExampleEndpoint:
    """Test the /api/examples/full_day GET endpoint."""

    def test_full_day_example_no_model(self, client):
        """Get full day example without model returns 24-hour example."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = None

            response = client.get("/api/examples/full_day")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "example" in data
            assert len(data["example"]["scenario_features"]) == 24
            assert len(data["example"]["timeslots"]) == 24
            assert data["example"]["update_historical"] is True
            assert data["model_available"] is False

    def test_full_day_example_with_model(self, client, mock_model):
        """Get full day example with model returns example with all features."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            response = client.get("/api/examples/full_day")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["model_available"] is True
            # Check that each hour slot contains required features
            for features in data["example"]["scenario_features"]:
                assert "outdoor_temp" in features
                assert "hour_of_day" in features
                assert "target_temp" in features

    def test_full_day_example_temperature_variation(self, client):
        """Full day example has realistic temperature variation."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = None

            response = client.get("/api/examples/full_day")

            data = response.get_json()
            features = data["example"]["scenario_features"]
            
            # Night hours should have lower target temp
            night_target = features[2]["target_temp"]  # 2 AM
            day_target = features[14]["target_temp"]  # 2 PM
            assert night_target < day_target

    def test_full_day_example_setpoint_schedule(self, client):
        """Full day example has realistic setpoint schedule."""
        with patch("app._get_model") as mock_get_model:
            mock_get_model.return_value = None

            response = client.get("/api/examples/full_day")

            data = response.get_json()
            features = data["example"]["scenario_features"]
            
            # Check is_night flag
            assert features[3]["is_night"] == 1  # 3 AM is night
            assert features[12]["is_night"] == 0  # Noon is not night


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
        """Resample with custom sample rate uses provided value."""
        mock_stats = ResampleStats(
            slots_processed=50,
            slots_saved=45,
            slots_skipped=5,
            categories=["outdoor_temp", "wind"],
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 20, 0, 0),
            sample_rate_minutes=10,
        )
        with patch("app.resample_all_categories") as mock_resample:
            mock_resample.return_value = mock_stats

            response = client.post(
                "/resample",
                json={"sample_rate_minutes": 10},
                content_type="application/json",
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["stats"]["sample_rate_minutes"] == 10
            mock_resample.assert_called_once_with(10, flush=False)

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
        """Resample without custom rate uses default."""
        mock_stats = ResampleStats(
            slots_processed=100,
            slots_saved=90,
            slots_skipped=10,
            categories=["outdoor_temp", "wind"],
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 20, 0, 0),
            sample_rate_minutes=5,
        )
        with patch("app.resample_all_categories") as mock_resample:
            mock_resample.return_value = mock_stats

            response = client.post("/resample")

            assert response.status_code == 200
            data = response.get_json()
            assert data["stats"]["sample_rate_minutes"] == 5
            mock_resample.assert_called_once()

    def test_resample_with_flush(self, client):
        """Resample with flush=true clears existing data."""
        mock_stats = ResampleStats(
            slots_processed=100,
            slots_saved=90,
            slots_skipped=10,
            categories=["outdoor_temp", "wind"],
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 20, 0, 0),
            sample_rate_minutes=10,
            table_flushed=True,
        )
        with patch("app.resample_all_categories") as mock_resample:
            mock_resample.return_value = mock_stats

            response = client.post(
                "/resample",
                json={"sample_rate_minutes": 10, "flush": True},
                content_type="application/json",
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["stats"]["sample_rate_minutes"] == 10
            assert data["stats"]["table_flushed"] is True
            mock_resample.assert_called_once_with(10, flush=True)

    def test_resample_table_flushed_in_response(self, client):
        """Resample response includes table_flushed field."""
        mock_stats = ResampleStats(
            slots_processed=50,
            slots_saved=40,
            slots_skipped=10,
            categories=["outdoor_temp"],
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 18, 0, 0),
            sample_rate_minutes=5,
            table_flushed=False,
        )
        with patch("app.resample_all_categories") as mock_resample:
            mock_resample.return_value = mock_stats

            response = client.post(
                "/resample",
                json={"sample_rate_minutes": 5},
                content_type="application/json",
            )

            assert response.status_code == 200
            data = response.get_json()
            assert "table_flushed" in data["stats"]
            assert data["stats"]["table_flushed"] is False

    def test_resample_with_flush_default_rate(self, client):
        """Resample with flush=true and default sample rate clears existing data."""
        mock_stats = ResampleStats(
            slots_processed=100,
            slots_saved=90,
            slots_skipped=10,
            categories=["outdoor_temp", "wind"],
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 20, 0, 0),
            sample_rate_minutes=5,
            table_flushed=True,
        )
        with patch("app.resample_all_categories") as mock_resample:
            mock_resample.return_value = mock_stats

            response = client.post(
                "/resample",
                json={"flush": True},
                content_type="application/json",
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["stats"]["table_flushed"] is True
            mock_resample.assert_called_once_with(None, flush=True)
