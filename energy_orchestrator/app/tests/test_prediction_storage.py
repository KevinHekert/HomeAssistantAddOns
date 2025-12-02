"""
Tests for prediction storage module.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os


@pytest.fixture(scope="module", autouse=True)
def setup_temp_dir():
    """Set up temp directory before module loads."""
    temp_dir = tempfile.mkdtemp()
    os.environ["DATA_DIR"] = temp_dir
    yield temp_dir


# Import after setting environment variable
@pytest.fixture
def storage_imports():
    """Import storage module after env is set."""
    import importlib
    import db.prediction_storage as storage
    importlib.reload(storage)
    return storage


class TestStoredPrediction:
    """Test StoredPrediction dataclass."""

    def test_creates_prediction(self, storage_imports):
        """Creates prediction with all fields."""
        pred = storage_imports.StoredPrediction(
            id="test_id",
            created_at="2024-12-02T14:00:00",
            source="weerlive",
            location="Amsterdam",
            timeslots=[{"timestamp": "2024-12-02T15:00:00"}],
            predictions=[{"predicted_kwh": 1.5}],
            total_kwh=1.5,
            model_type="single_step",
        )
        assert pred.id == "test_id"
        assert pred.source == "weerlive"
        assert pred.total_kwh == 1.5


class TestStorePrediction:
    """Test store_prediction function."""

    def test_stores_prediction(self, storage_imports):
        """Stores prediction successfully."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        timeslots = [{"timestamp": "2024-12-02T14:00:00", "outdoor_temperature": 10.0}]
        predictions = [{"timestamp": "2024-12-02T14:00:00", "predicted_kwh": 1.5}]
        
        success, error, pred_id = storage_imports.store_prediction(
            timeslots=timeslots,
            predictions=predictions,
            total_kwh=1.5,
            source="manual",
        )
        
        assert success is True
        assert error is None
        assert pred_id is not None

    def test_stores_with_metadata(self, storage_imports):
        """Stores prediction with source and location."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        success, error, pred_id = storage_imports.store_prediction(
            timeslots=[],
            predictions=[{"predicted_kwh": 1.0}],
            total_kwh=1.0,
            source="weerlive",
            location="Amsterdam",
            model_type="two_step",
        )
        
        assert success is True
        
        pred = storage_imports.get_prediction_by_id(pred_id)
        assert pred.source == "weerlive"
        assert pred.location == "Amsterdam"
        assert pred.model_type == "two_step"

    def test_generates_unique_ids(self, storage_imports):
        """Generates unique IDs for each prediction."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        ids = set()
        for _ in range(5):
            success, error, pred_id = storage_imports.store_prediction(
                timeslots=[],
                predictions=[{"predicted_kwh": 1.0}],
                total_kwh=1.0,
            )
            assert success
            ids.add(pred_id)
        
        assert len(ids) == 5


class TestGetStoredPredictions:
    """Test get_stored_predictions function."""

    def test_returns_empty_when_no_predictions(self, storage_imports):
        """Returns empty list when no predictions stored."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        predictions = storage_imports.get_stored_predictions()
        assert predictions == []

    def test_returns_stored_predictions(self, storage_imports):
        """Returns all stored predictions."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        storage_imports.store_prediction(timeslots=[], predictions=[{"predicted_kwh": 1.0}], total_kwh=1.0)
        storage_imports.store_prediction(timeslots=[], predictions=[{"predicted_kwh": 2.0}], total_kwh=2.0)
        
        predictions = storage_imports.get_stored_predictions()
        assert len(predictions) == 2

    def test_returns_newest_first(self, storage_imports):
        """Returns predictions newest first."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        storage_imports.store_prediction(timeslots=[], predictions=[{"predicted_kwh": 1.0}], total_kwh=1.0)
        storage_imports.store_prediction(timeslots=[], predictions=[{"predicted_kwh": 2.0}], total_kwh=2.0)
        
        predictions = storage_imports.get_stored_predictions()
        # Second one stored should be first in list (newest first)
        assert predictions[0].total_kwh == 2.0


class TestGetPredictionById:
    """Test get_prediction_by_id function."""

    def test_returns_none_for_nonexistent(self, storage_imports):
        """Returns None for nonexistent ID."""
        pred = storage_imports.get_prediction_by_id("nonexistent")
        assert pred is None

    def test_returns_prediction(self, storage_imports):
        """Returns prediction by ID."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        success, error, pred_id = storage_imports.store_prediction(
            timeslots=[{"ts": 1}],
            predictions=[{"predicted_kwh": 1.5}],
            total_kwh=1.5,
        )
        
        pred = storage_imports.get_prediction_by_id(pred_id)
        assert pred is not None
        assert pred.id == pred_id
        assert pred.total_kwh == 1.5


class TestDeletePrediction:
    """Test delete_prediction function."""

    def test_deletes_prediction(self, storage_imports):
        """Deletes prediction by ID."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        success, error, pred_id = storage_imports.store_prediction(
            timeslots=[],
            predictions=[{"predicted_kwh": 1.0}],
            total_kwh=1.0,
        )
        
        deleted = storage_imports.delete_prediction(pred_id)
        assert deleted is True
        
        pred = storage_imports.get_prediction_by_id(pred_id)
        assert pred is None

    def test_returns_false_for_nonexistent(self, storage_imports):
        """Returns False for nonexistent ID."""
        deleted = storage_imports.delete_prediction("nonexistent")
        assert deleted is False


class TestGetPredictionListSummary:
    """Test get_prediction_list_summary function."""

    def test_returns_empty_for_no_predictions(self, storage_imports):
        """Returns empty list when no predictions."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        summaries = storage_imports.get_prediction_list_summary()
        assert summaries == []

    def test_returns_summaries(self, storage_imports):
        """Returns summary list for predictions."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        storage_imports.store_prediction(
            timeslots=[],
            predictions=[
                {"timestamp": "2024-12-02T14:00:00", "predicted_kwh": 1.0},
                {"timestamp": "2024-12-02T15:00:00", "predicted_kwh": 1.5},
            ],
            total_kwh=2.5,
            source="weerlive",
            location="Amsterdam",
        )
        
        summaries = storage_imports.get_prediction_list_summary()
        assert len(summaries) == 1
        
        summary = summaries[0]
        assert summary["source"] == "weerlive"
        assert summary["location"] == "Amsterdam"
        assert summary["total_kwh"] == 2.5
        assert summary["slots_count"] == 2
        assert summary["start_time"] == "2024-12-02T14:00:00"
        assert summary["end_time"] == "2024-12-02T15:00:00"


class TestLoadSavePredictions:
    """Test _load_predictions and _save_predictions functions."""

    def test_loads_empty_when_file_missing(self, storage_imports):
        """Returns empty list when file doesn't exist."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        predictions = storage_imports._load_predictions()
        assert predictions == []

    def test_saves_and_loads(self, storage_imports):
        """Saves and loads predictions correctly."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        data = [
            {"id": "1", "total_kwh": 1.0},
            {"id": "2", "total_kwh": 2.0},
        ]
        
        storage_imports._save_predictions(data)
        loaded = storage_imports._load_predictions()
        
        assert len(loaded) == 2
        assert loaded[0]["id"] == "1"

    def test_limits_stored_predictions(self, storage_imports):
        """Limits number of stored predictions."""
        # Clean up first
        if storage_imports.PREDICTIONS_FILE_PATH.exists():
            storage_imports.PREDICTIONS_FILE_PATH.unlink()
        
        # Create more than MAX_STORED_PREDICTIONS
        data = [{"id": str(i), "total_kwh": float(i)} for i in range(150)]
        
        storage_imports._save_predictions(data)
        loaded = storage_imports._load_predictions()
        
        # Should be limited to MAX_STORED_PREDICTIONS (100)
        assert len(loaded) == 100
        # Should keep the most recent ones
        assert loaded[0]["id"] == "50"  # First 50 were dropped


class TestComparisonResult:
    """Test ComparisonResult dataclass."""

    def test_creates_comparison_with_actual(self, storage_imports):
        """Creates comparison result with actual data."""
        result = storage_imports.ComparisonResult(
            prediction_id="test",
            timestamp="2024-12-02T14:00:00",
            predicted_kwh=1.5,
            actual_kwh=1.3,
            delta_kwh=0.2,
            delta_pct=15.4,
            has_actual=True,
        )
        assert result.has_actual is True
        assert result.delta_kwh == 0.2

    def test_creates_comparison_without_actual(self, storage_imports):
        """Creates comparison result without actual data."""
        result = storage_imports.ComparisonResult(
            prediction_id="test",
            timestamp="2024-12-02T14:00:00",
            predicted_kwh=1.5,
            actual_kwh=None,
            delta_kwh=None,
            delta_pct=None,
            has_actual=False,
        )
        assert result.has_actual is False
        assert result.actual_kwh is None
