"""
Tests for Flask app endpoints.
"""

import pytest
from unittest.mock import patch

from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


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
