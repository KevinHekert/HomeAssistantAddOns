"""
Tests for optimizer storage and async execution functionality.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

from app import app
from db.optimizer_storage import (
    save_optimizer_run,
    get_latest_optimizer_run,
    get_optimizer_run_by_id,
    get_optimizer_result_by_id,
    list_optimizer_runs,
)
from db import OptimizerRun, OptimizerResult
from ml.optimizer import OptimizerProgress, OptimizationResult


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_optimization_result():
    """Create a sample optimization result."""
    return OptimizationResult(
        config_name="Test Config",
        model_type="single_step",
        experimental_features={"pressure": True, "outdoor_temp_avg_6h": False},
        val_mape_pct=10.5,
        val_mae_kwh=0.15,
        val_r2=0.85,
        train_samples=60,
        val_samples=20,
        success=True,
        training_timestamp=datetime.now(),
    )


@pytest.fixture
def sample_optimizer_progress(sample_optimization_result):
    """Create a sample optimizer progress object."""
    progress = OptimizerProgress(
        total_configurations=2,
        completed_configurations=2,
        current_configuration="",
        current_model_type="",
        phase="complete",
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    progress.results = [sample_optimization_result]
    progress.best_result = sample_optimization_result
    return progress


class TestOptimizerStorage:
    """Test optimizer storage functions."""

    def test_save_optimizer_run_success(self, sample_optimizer_progress):
        """Save optimizer run stores data in database."""
        with patch("db.optimizer_storage.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__enter__.return_value = mock_session
            
            # Mock the flush to set IDs
            def mock_flush():
                # Simulate DB assigning ID
                for call in mock_session.add.call_args_list:
                    obj = call[0][0]
                    if isinstance(obj, OptimizerRun):
                        obj.id = 1
                    elif isinstance(obj, OptimizerResult):
                        obj.id = 100
            
            mock_session.flush.side_effect = mock_flush
            
            run_id = save_optimizer_run(sample_optimizer_progress)
            
            assert run_id == 1
            # Verify session.add was called for run and result
            assert mock_session.add.call_count >= 2
            mock_session.commit.assert_called_once()
    
    def test_save_optimizer_run_handles_error(self, sample_optimizer_progress):
        """Save optimizer run handles database errors gracefully."""
        with patch("db.optimizer_storage.Session") as mock_session_class:
            mock_session_class.return_value.__enter__.return_value.add.side_effect = Exception("DB Error")
            
            run_id = save_optimizer_run(sample_optimizer_progress)
            
            assert run_id is None
    
    def test_get_latest_optimizer_run_returns_most_recent(self):
        """Get latest optimizer run returns the most recent run."""
        with patch("db.optimizer_storage.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__enter__.return_value = mock_session
            
            # Mock the run
            mock_run = MagicMock()
            mock_run.id = 1
            mock_run.start_time = datetime.now()
            mock_run.end_time = datetime.now()
            mock_run.phase = "complete"
            mock_run.total_configurations = 2
            mock_run.completed_configurations = 2
            mock_run.best_result_id = 100
            mock_run.error_message = None
            
            # Mock the result
            mock_result = MagicMock()
            mock_result.id = 100
            mock_result.run_id = 1
            mock_result.config_name = "Test"
            mock_result.model_type = "single_step"
            mock_result.experimental_features_json = json.dumps({"pressure": True})
            mock_result.val_mape_pct = 10.5
            mock_result.val_mae_kwh = 0.15
            mock_result.val_r2 = 0.85
            mock_result.train_samples = 60
            mock_result.val_samples = 20
            mock_result.success = True
            mock_result.error_message = None
            mock_result.training_timestamp = datetime.now()
            
            mock_session.scalars.return_value.first.return_value = mock_run
            mock_session.scalars.return_value.all.return_value = [mock_result]
            
            run = get_latest_optimizer_run()
            
            assert run is not None
            assert run["id"] == 1
            assert run["phase"] == "complete"
            assert len(run["results"]) == 1
    
    def test_get_latest_optimizer_run_returns_none_when_no_runs(self):
        """Get latest optimizer run returns None when database is empty."""
        with patch("db.optimizer_storage.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__enter__.return_value = mock_session
            mock_session.scalars.return_value.first.return_value = None
            
            run = get_latest_optimizer_run()
            
            assert run is None
    
    def test_get_optimizer_run_by_id_returns_correct_run(self):
        """Get optimizer run by ID returns the correct run."""
        with patch("db.optimizer_storage.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__enter__.return_value = mock_session
            
            # Mock the run
            mock_run = MagicMock()
            mock_run.id = 5
            mock_run.start_time = datetime.now()
            mock_run.phase = "complete"
            mock_run.best_result_id = None
            
            mock_session.get.return_value = mock_run
            mock_session.scalars.return_value.all.return_value = []
            
            run = get_optimizer_run_by_id(5)
            
            assert run is not None
            assert run["id"] == 5
            mock_session.get.assert_called_once_with(OptimizerRun, 5)
    
    def test_get_optimizer_result_by_id_returns_correct_result(self):
        """Get optimizer result by ID returns the correct result."""
        with patch("db.optimizer_storage.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__enter__.return_value = mock_session
            
            # Mock the result
            mock_result = MagicMock()
            mock_result.id = 100
            mock_result.run_id = 1
            mock_result.config_name = "Test Config"
            mock_result.model_type = "two_step"
            mock_result.experimental_features_json = json.dumps({"pressure": False})
            mock_result.val_mape_pct = 8.0
            mock_result.val_mae_kwh = 0.18
            mock_result.val_r2 = 0.85
            mock_result.train_samples = 200
            mock_result.val_samples = 50
            mock_result.success = True
            mock_result.error_message = None
            mock_result.training_timestamp = None
            
            mock_session.get.return_value = mock_result
            
            result = get_optimizer_result_by_id(100)
            
            assert result is not None
            assert result["id"] == 100
            assert result["config_name"] == "Test Config"
            assert result["val_mape_pct"] == 8.0
            mock_session.get.assert_called_once_with(OptimizerResult, 100)
    
    def test_list_optimizer_runs_returns_summaries(self):
        """List optimizer runs returns run summaries."""
        with patch("db.optimizer_storage.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__enter__.return_value = mock_session
            
            # Mock two runs
            mock_run1 = MagicMock()
            mock_run1.id = 1
            mock_run1.start_time = datetime.now()
            mock_run1.end_time = datetime.now()
            mock_run1.phase = "complete"
            mock_run1.total_configurations = 2
            mock_run1.completed_configurations = 2
            mock_run1.best_result_id = None
            
            mock_run2 = MagicMock()
            mock_run2.id = 2
            mock_run2.start_time = datetime.now()
            mock_run2.end_time = None
            mock_run2.phase = "training"
            mock_run2.total_configurations = 4
            mock_run2.completed_configurations = 1
            mock_run2.best_result_id = None
            
            mock_session.scalars.return_value.all.return_value = [mock_run2, mock_run1]
            
            runs = list_optimizer_runs(limit=10)
            
            assert len(runs) == 2
            assert runs[0]["id"] == 2
            assert runs[1]["id"] == 1


class TestOptimizerEndpointsAsync:
    """Test optimizer API endpoints with async functionality."""

    def test_run_optimizer_starts_background_thread(self, client):
        """Run optimizer starts optimization in background thread."""
        with patch("app.threading.Thread") as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread_class.return_value = mock_thread
            
            response = client.post("/api/optimizer/run")
            
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["running"] is True
            assert "background" in data["message"].lower()
            
            # Verify thread was started
            mock_thread_class.assert_called_once()
            mock_thread.start.assert_called_once()
    
    def test_run_optimizer_rejects_concurrent_runs(self, client):
        """Run optimizer rejects request when already running."""
        import app as app_module
        app_module._optimizer_running = True
        
        try:
            response = client.post("/api/optimizer/run")
            
            assert response.status_code == 400
            data = response.get_json()
            assert data["status"] == "error"
            assert "already running" in data["message"].lower()
        finally:
            app_module._optimizer_running = False
    
    def test_get_optimizer_status_returns_results_when_complete(self, client):
        """Get optimizer status returns full results when optimization is complete."""
        import app as app_module
        from ml.optimizer import OptimizerProgress, OptimizationResult
        
        # Set up completed progress
        result = OptimizationResult(
            config_name="Test",
            model_type="single_step",
            experimental_features={"pressure": True},
            val_mape_pct=10.0,
            val_mae_kwh=0.15,
            val_r2=0.85,
            train_samples=60,
            val_samples=20,
            success=True,
        )
        
        progress = OptimizerProgress(
            total_configurations=2,
            completed_configurations=2,
            current_configuration="",
            current_model_type="",
            phase="complete",
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        progress.results = [result]
        progress.best_result = result
        
        app_module._optimizer_progress = progress
        app_module._optimizer_running = False
        
        try:
            response = client.get("/api/optimizer/status")
            
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["running"] is False
            assert data["progress"]["phase"] == "complete"
            assert "results" in data["progress"]
            assert len(data["progress"]["results"]) == 1
        finally:
            app_module._optimizer_progress = None
    
    def test_apply_optimizer_result_by_id_success(self, client):
        """Apply optimizer result by ID applies configuration."""
        with patch("app.get_optimizer_result_by_id") as mock_get_result, \
             patch("app.apply_best_configuration") as mock_apply:
            
            mock_get_result.return_value = {
                "id": 100,
                "run_id": 1,
                "config_name": "Test",
                "model_type": "single_step",
                "experimental_features": {"pressure": True},
                "val_mape_pct": 10.0,
                "val_mae_kwh": 0.15,
                "val_r2": 0.85,
                "train_samples": 60,
                "val_samples": 20,
                "success": True,
                "error_message": None,
            }
            mock_apply.return_value = True
            
            response = client.post(
                "/api/optimizer/apply/100",
                json={"enable_two_step": True},
                content_type="application/json",
            )
            
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert "applied_settings" in data
            mock_apply.assert_called_once()
    
    def test_apply_optimizer_result_by_id_not_found(self, client):
        """Apply optimizer result by ID returns 404 for nonexistent result."""
        with patch("app.get_optimizer_result_by_id") as mock_get_result:
            mock_get_result.return_value = None
            
            response = client.post(
                "/api/optimizer/apply/999",
                json={},
                content_type="application/json",
            )
            
            assert response.status_code == 404
            data = response.get_json()
            assert data["status"] == "error"
            assert "not found" in data["message"].lower()
    
    def test_get_optimizer_runs_returns_list(self, client):
        """Get optimizer runs returns list of runs."""
        with patch("app.list_optimizer_runs") as mock_list_runs:
            mock_list_runs.return_value = [
                {
                    "id": 1,
                    "start_time": datetime.now().isoformat(),
                    "phase": "complete",
                    "total_configurations": 2,
                },
            ]
            
            response = client.get("/api/optimizer/runs")
            
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert len(data["runs"]) == 1
    
    def test_get_optimizer_latest_returns_most_recent(self, client):
        """Get latest optimizer run returns most recent run."""
        with patch("app.get_latest_optimizer_run") as mock_get_latest:
            mock_get_latest.return_value = {
                "id": 5,
                "start_time": datetime.now().isoformat(),
                "phase": "complete",
                "results": [],
            }
            
            response = client.get("/api/optimizer/latest")
            
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "success"
            assert data["run"]["id"] == 5
