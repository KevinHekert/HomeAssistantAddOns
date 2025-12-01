"""
Tests for Flask app endpoints.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd

from app import app
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
        """Successful resample returns 200 with success message."""
        with patch("app.resample_all_categories_to_5min") as mock_resample:
            response = client.post("/resample")

            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "successfully" in data["message"]
            mock_resample.assert_called_once()

    def test_resample_error(self, client):
        """Error during resample returns 500 with error message."""
        with patch("app.resample_all_categories_to_5min") as mock_resample:
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
