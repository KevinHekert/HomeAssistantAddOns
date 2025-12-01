"""
Tests for the heating features ETL module.

Uses an in-memory SQLite database to test:
1. Feature dataset construction
2. Historical aggregation features
3. Target computation
4. Handling of limited data
"""

import pytest
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db import Base, ResampledSample
from ml.heating_features import (
    _load_resampled_data,
    _pivot_data,
    _compute_historical_aggregations,
    _compute_target,
    _add_time_features,
    build_heating_feature_dataset,
)
import db.core as core_module
import ml.heating_features as heating_features_module


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def patch_engine(test_engine, monkeypatch):
    """Patch the engine in all relevant modules."""
    monkeypatch.setattr(core_module, "engine", test_engine)
    monkeypatch.setattr(heating_features_module, "engine", test_engine)
    return test_engine


def _create_resampled_samples(session: Session, start_time: datetime, num_slots: int):
    """
    Create test resampled samples with all required categories.
    
    Creates data for 5-minute slots with realistic values.
    """
    categories = {
        "outdoor_temp": (5.0, 0.1),  # base, increment
        "wind": (3.0, 0.05),
        "humidity": (75.0, -0.1),
        "pressure": (1013.0, 0.01),
        "indoor_temp": (20.0, 0.02),
        "target_temp": (21.0, 0),
        "hp_kwh_total": (1000.0, 0.1),  # Cumulative, increment each slot
        "dhw_active": (0.0, 0),  # No DHW activity
    }
    
    for i in range(num_slots):
        slot_start = start_time + timedelta(minutes=5 * i)
        
        for category, (base, increment) in categories.items():
            value = base + increment * i
            session.add(ResampledSample(
                slot_start=slot_start,
                category=category,
                value=value,
                unit="test",
            ))
    
    session.commit()


class TestLoadResampledData:
    """Test the _load_resampled_data function."""
    
    def test_empty_database(self, patch_engine):
        """Empty database returns empty DataFrame."""
        with Session(patch_engine) as session:
            df = _load_resampled_data(session)
        
        assert df.empty
        assert list(df.columns) == ["slot_start", "category", "value"]
    
    def test_loads_data_correctly(self, patch_engine):
        """Data is loaded correctly from database."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        
        with Session(patch_engine) as session:
            session.add(ResampledSample(
                slot_start=start,
                category="outdoor_temp",
                value=10.0,
                unit="°C",
            ))
            session.add(ResampledSample(
                slot_start=start,
                category="wind",
                value=5.0,
                unit="m/s",
            ))
            session.commit()
            
            df = _load_resampled_data(session)
        
        assert len(df) == 2
        assert set(df["category"]) == {"outdoor_temp", "wind"}


class TestPivotData:
    """Test the _pivot_data function."""
    
    def test_empty_dataframe(self):
        """Empty DataFrame returns empty result."""
        df = pd.DataFrame(columns=["slot_start", "category", "value"])
        result = _pivot_data(df)
        assert result.empty
    
    def test_pivots_correctly(self):
        """Data is pivoted correctly."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        df = pd.DataFrame([
            {"slot_start": start, "category": "outdoor_temp", "value": 10.0},
            {"slot_start": start, "category": "wind", "value": 5.0},
        ])
        
        result = _pivot_data(df)
        
        assert len(result) == 1
        assert result.loc[start, "outdoor_temp"] == 10.0
        assert result.loc[start, "wind"] == 5.0


class TestComputeHistoricalAggregations:
    """Test the _compute_historical_aggregations function."""
    
    def test_short_term_averages(self):
        """Short-term averages are computed correctly."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        # Create 24 slots (2 hours) of data
        slots = [start + timedelta(minutes=5 * i) for i in range(24)]
        
        df = pd.DataFrame(
            {"outdoor_temp": [10.0 + i * 0.1 for i in range(24)]},
            index=pd.DatetimeIndex(slots),
        )
        
        result = _compute_historical_aggregations(df, available_history_hours=2.0)
        
        assert "outdoor_temp_avg_1h" in result.columns
        # 1h average should be computed
        assert not result["outdoor_temp_avg_1h"].isna().all()
    
    def test_no_7d_features_with_short_history(self):
        """7-day features are not added with insufficient history."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        slots = [start + timedelta(minutes=5 * i) for i in range(24)]
        
        df = pd.DataFrame(
            {"outdoor_temp": [10.0] * 24},
            index=pd.DatetimeIndex(slots),
        )
        
        result = _compute_historical_aggregations(df, available_history_hours=2.0)
        
        assert "outdoor_temp_avg_7d" not in result.columns
    
    def test_7d_features_with_long_history(self):
        """7-day features are added with sufficient history."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        # Create enough slots for 7+ days
        num_slots = 24 * 7 * 12 + 100  # More than 7 days
        slots = [start + timedelta(minutes=5 * i) for i in range(num_slots)]
        
        df = pd.DataFrame(
            {"outdoor_temp": [10.0] * num_slots},
            index=pd.DatetimeIndex(slots),
        )
        
        result = _compute_historical_aggregations(df, available_history_hours=170.0)
        
        assert "outdoor_temp_avg_7d" in result.columns


class TestComputeTarget:
    """Test the _compute_target function."""
    
    def test_computes_target_correctly(self):
        """Target is computed as forward kWh delta."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        # Create slots for 2 hours
        num_slots = 24
        slots = [start + timedelta(minutes=5 * i) for i in range(num_slots)]
        
        # Linear increase: 0.1 kWh per slot
        df = pd.DataFrame(
            {"hp_kwh_total": [100.0 + i * 0.1 for i in range(num_slots)]},
            index=pd.DatetimeIndex(slots),
        )
        
        result = _compute_target(df, horizon_slots=12)
        
        assert "target_heating_kwh_1h" in result.columns
        # First slot should have target = 12 * 0.1 = 1.2 kWh
        assert abs(result.iloc[0]["target_heating_kwh_1h"] - 1.2) < 0.001
    
    def test_excludes_dhw_slots(self):
        """Slots with DHW activity are excluded from target."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        num_slots = 24
        slots = [start + timedelta(minutes=5 * i) for i in range(num_slots)]
        
        # Create data with DHW active at slot 5
        dhw_active = [0.0] * num_slots
        dhw_active[5] = 1.0
        
        df = pd.DataFrame({
            "hp_kwh_total": [100.0 + i * 0.1 for i in range(num_slots)],
            "dhw_active": dhw_active,
        }, index=pd.DatetimeIndex(slots))
        
        result = _compute_target(df, horizon_slots=12)
        
        # Slot 0's horizon includes DHW at slot 5, so target should be None
        assert pd.isna(result.iloc[0]["target_heating_kwh_1h"])


class TestAddTimeFeatures:
    """Test the _add_time_features function."""
    
    def test_adds_time_features(self):
        """Time features are added correctly."""
        # Monday at 14:00
        start = datetime(2024, 1, 1, 14, 0, 0)  # Monday
        slots = [start + timedelta(minutes=5 * i) for i in range(3)]
        
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.DatetimeIndex(slots),
        )
        
        result = _add_time_features(df)
        
        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "is_night" in result.columns
        
        assert result.iloc[0]["hour_of_day"] == 14
        assert result.iloc[0]["day_of_week"] == 0  # Monday
        assert result.iloc[0]["is_weekend"] == 0
        assert result.iloc[0]["is_night"] == 0
    
    def test_night_feature(self):
        """Night feature is set correctly for night hours."""
        start = datetime(2024, 1, 1, 23, 0, 0)  # 11 PM
        slots = [start + timedelta(minutes=5 * i) for i in range(3)]
        
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.DatetimeIndex(slots),
        )
        
        result = _add_time_features(df)
        
        assert result.iloc[0]["is_night"] == 1


class TestBuildHeatingFeatureDataset:
    """Test the main build_heating_feature_dataset function."""
    
    def test_empty_database(self, patch_engine):
        """Empty database returns None with stats."""
        df, stats = build_heating_feature_dataset(min_samples=10)
        
        assert df is None
        assert stats.total_slots == 0
        assert stats.valid_slots == 0
    
    def test_insufficient_samples(self, patch_engine):
        """Returns None if below min_samples threshold."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        
        with Session(patch_engine) as session:
            # Create only 5 slots - not enough
            _create_resampled_samples(session, start, num_slots=5)
        
        df, stats = build_heating_feature_dataset(min_samples=100)
        
        assert df is None
        assert stats.total_slots == 5
    
    def test_builds_dataset_with_sufficient_data(self, patch_engine):
        """Builds complete dataset with sufficient data."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        
        with Session(patch_engine) as session:
            # Create enough data for training (200 slots = ~17 hours)
            _create_resampled_samples(session, start, num_slots=200)
        
        df, stats = build_heating_feature_dataset(min_samples=50)
        
        assert df is not None
        assert stats.valid_slots > 0
        assert len(stats.features_used) > 0
        
        # Check that no HP output features are in the dataset
        for feat in stats.features_used:
            assert "flow_temp" not in feat
            assert "return_temp" not in feat
            assert "hp_power" not in feat.lower()
        
        # Target column should be present
        assert "target_heating_kwh_1h" in df.columns
    
    def test_logs_features_used(self, patch_engine, caplog):
        """Logs information about features used."""
        import logging
        caplog.set_level(logging.INFO)
        
        start = datetime(2024, 1, 1, 12, 0, 0)
        
        with Session(patch_engine) as session:
            _create_resampled_samples(session, start, num_slots=200)
        
        df, stats = build_heating_feature_dataset(min_samples=50)
        
        # Should log features
        assert any("Features used" in record.message for record in caplog.records)
