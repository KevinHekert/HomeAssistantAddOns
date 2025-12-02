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
        """Time features are added correctly (without timezone conversion)."""
        # Monday at 14:00
        start = datetime(2024, 1, 1, 14, 0, 0)  # Monday
        slots = [start + timedelta(minutes=5 * i) for i in range(3)]
        
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.DatetimeIndex(slots),
        )
        
        # Use use_configured_timezone=False for deterministic test
        result = _add_time_features(df, use_configured_timezone=False)
        
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
        
        # Use use_configured_timezone=False for deterministic test
        result = _add_time_features(df, use_configured_timezone=False)
        
        assert result.iloc[0]["is_night"] == 1
    
    def test_with_timezone_conversion(self):
        """Test that timezone conversion works correctly."""
        # Create a UTC timestamp (winter time - CET is UTC+1)
        start = datetime(2024, 1, 1, 13, 0, 0)  # 13:00 UTC = 14:00 CET
        slots = [start + timedelta(minutes=5 * i) for i in range(3)]
        
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.DatetimeIndex(slots),
        )
        
        # Default timezone is Europe/Amsterdam (CET = UTC+1 in winter)
        result = _add_time_features(df, use_configured_timezone=True)
        
        # 13:00 UTC = 14:00 in Amsterdam during winter
        assert result.iloc[0]["hour_of_day"] == 14


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


class TestComputeScenarioHistoricalFeatures:
    """Test the compute_scenario_historical_features function."""
    
    def test_empty_scenario(self):
        """Empty scenario returns empty list."""
        from ml.heating_features import compute_scenario_historical_features
        
        result = compute_scenario_historical_features([])
        assert result == []
    
    def test_single_slot_scenario(self):
        """Single slot scenario adds historical features."""
        from ml.heating_features import compute_scenario_historical_features
        
        scenario = [{
            "outdoor_temp": 5.0,
            "indoor_temp": 20.0,
            "target_temp": 21.0,
            "wind": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
        }]
        
        result = compute_scenario_historical_features(scenario)
        
        assert len(result) == 1
        # Check that historical features are added
        assert "outdoor_temp_avg_1h" in result[0]
        assert "outdoor_temp_avg_6h" in result[0]
        assert "outdoor_temp_avg_24h" in result[0]
        assert "indoor_temp_avg_6h" in result[0]
        assert "target_temp_avg_6h" in result[0]
        assert "heating_degree_hours_24h" in result[0]
        
        # For single slot, 1h avg should equal current value
        assert result[0]["outdoor_temp_avg_1h"] == 5.0
    
    def test_multi_slot_scenario_computes_rolling(self):
        """Multi-slot scenario computes rolling averages."""
        from ml.heating_features import compute_scenario_historical_features
        
        # 6 hourly slots with varying outdoor temp
        scenario = [
            {"outdoor_temp": 5.0, "indoor_temp": 20.0, "target_temp": 21.0},
            {"outdoor_temp": 4.0, "indoor_temp": 20.0, "target_temp": 21.0},
            {"outdoor_temp": 3.0, "indoor_temp": 20.0, "target_temp": 21.0},
            {"outdoor_temp": 4.0, "indoor_temp": 20.0, "target_temp": 21.0},
            {"outdoor_temp": 5.0, "indoor_temp": 20.0, "target_temp": 21.0},
            {"outdoor_temp": 6.0, "indoor_temp": 20.0, "target_temp": 21.0},
        ]
        
        result = compute_scenario_historical_features(scenario)
        
        assert len(result) == 6
        
        # Last slot should have 6h avg = mean of all 6 values
        expected_avg = (5.0 + 4.0 + 3.0 + 4.0 + 5.0 + 6.0) / 6
        assert abs(result[-1]["outdoor_temp_avg_6h"] - expected_avg) < 0.01
    
    def test_with_timeslots_adds_time_features(self):
        """Timeslots are used to compute time features."""
        from ml.heating_features import compute_scenario_historical_features
        
        scenario = [{"outdoor_temp": 5.0, "target_temp": 20.0}]
        timeslots = [datetime(2024, 1, 15, 14, 0, 0)]  # Monday 2pm
        
        result = compute_scenario_historical_features(scenario, timeslots=timeslots)
        
        assert len(result) == 1
        assert result[0]["hour_of_day"] == 14
        assert result[0]["day_of_week"] == 0  # Monday
        assert result[0]["is_weekend"] == 0
        assert result[0]["is_night"] == 0
    
    def test_preserves_existing_features(self):
        """Existing features are not overwritten."""
        from ml.heating_features import compute_scenario_historical_features
        
        scenario = [{
            "outdoor_temp": 5.0,
            "outdoor_temp_avg_1h": 10.0,  # Should be preserved
            "target_temp": 20.0,
        }]
        
        result = compute_scenario_historical_features(scenario)
        
        assert result[0]["outdoor_temp_avg_1h"] == 10.0  # Preserved, not recomputed
    
    def test_heating_degree_hours_computation(self):
        """Heating degree hours are computed correctly."""
        from ml.heating_features import compute_scenario_historical_features
        
        # 3 hours with constant temps
        scenario = [
            {"outdoor_temp": 10.0, "target_temp": 20.0},  # 10 degree diff
            {"outdoor_temp": 10.0, "target_temp": 20.0},  # 10 degree diff  
            {"outdoor_temp": 10.0, "target_temp": 20.0},  # 10 degree diff
        ]
        
        result = compute_scenario_historical_features(scenario)
        
        # For 3 hourly slots: sum of degree diffs * 1 hour each = 30
        assert result[-1]["heating_degree_hours_24h"] == 30.0


class TestGetActualVsPredictedData:
    """Test the get_actual_vs_predicted_data function."""
    
    def test_no_data_returns_none(self, patch_engine):
        """Returns None when no data is available."""
        from ml.heating_features import get_actual_vs_predicted_data
        
        start = datetime(2024, 1, 15, 12, 0, 0)
        end = datetime(2024, 1, 15, 18, 0, 0)
        
        df, error = get_actual_vs_predicted_data(start, end)
        
        assert df is None
        assert error is not None
    
    def test_returns_data_for_valid_range(self, patch_engine):
        """Returns data for a valid time range."""
        from ml.heating_features import get_actual_vs_predicted_data
        
        start = datetime(2024, 1, 15, 12, 0, 0)
        
        # Create some test data
        with Session(patch_engine) as session:
            for i in range(24):  # 2 hours of 5-min data
                slot_start = start + timedelta(minutes=5 * i)
                session.add(ResampledSample(
                    slot_start=slot_start,
                    category="outdoor_temp",
                    value=5.0 + i * 0.1,
                    unit="°C",
                ))
                session.add(ResampledSample(
                    slot_start=slot_start,
                    category="hp_kwh_total",
                    value=100.0 + i * 0.1,
                    unit="kWh",
                ))
            session.commit()
        
        end = start + timedelta(hours=2)
        df, error = get_actual_vs_predicted_data(start, end)
        
        assert df is not None
        assert error is None
        assert "outdoor_temp" in df.columns
        assert "actual_heating_kwh" in df.columns


class TestValidatePredictionStartTime:
    """Test the validate_prediction_start_time function."""
    
    def test_valid_next_hour(self):
        """Next hour is valid."""
        from ml.heating_features import validate_prediction_start_time
        
        next_hour = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        is_valid, message = validate_prediction_start_time(next_hour)
        
        assert is_valid
        assert "Valid" in message
    
    def test_valid_future_hour(self):
        """Future hours are valid."""
        from ml.heating_features import validate_prediction_start_time
        
        future = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=5)
        
        is_valid, message = validate_prediction_start_time(future)
        
        assert is_valid
    
    def test_invalid_past_time(self):
        """Past time is invalid."""
        from ml.heating_features import validate_prediction_start_time
        
        past = datetime.now() - timedelta(hours=2)
        past = past.replace(minute=0, second=0, microsecond=0)
        
        is_valid, message = validate_prediction_start_time(past)
        
        assert not is_valid
        assert "past" in message.lower() or "current" in message.lower()
    
    def test_invalid_non_aligned_time(self):
        """Non-hour-aligned time is invalid."""
        from ml.heating_features import validate_prediction_start_time
        
        next_hour = datetime.now().replace(second=0, microsecond=0) + timedelta(hours=1)
        non_aligned = next_hour.replace(minute=30)  # Half hour
        
        is_valid, message = validate_prediction_start_time(non_aligned)
        
        assert not is_valid
        assert "boundary" in message.lower()


class TestFeatureDatasetStatsTimeRange:
    """Test that FeatureDatasetStats includes time range info."""
    
    def test_stats_include_time_range(self, patch_engine):
        """Stats include data time range information."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        
        with Session(patch_engine) as session:
            _create_resampled_samples(session, start, num_slots=200)
        
        df, stats = build_heating_feature_dataset(min_samples=50)
        
        assert df is not None
        assert stats.data_start_time is not None
        assert stats.data_end_time is not None
        assert stats.available_history_hours is not None
        
        # Check time range makes sense
        assert stats.data_start_time < stats.data_end_time
        assert stats.available_history_hours > 0


class TestValidateSimplifiedScenario:
    """Tests for validate_simplified_scenario function."""
    
    def test_empty_timeslots(self):
        """Empty timeslots list returns invalid."""
        from ml.heating_features import validate_simplified_scenario
        
        result = validate_simplified_scenario([])
        
        assert not result.valid
        assert any("at least one" in err for err in result.errors)
    
    def test_valid_scenario(self):
        """Valid scenario with all required fields passes."""
        from ml.heating_features import validate_simplified_scenario
        
        next_hour = datetime.now() + timedelta(hours=1)
        timeslots = [{
            "timestamp": next_hour.isoformat(),
            "outdoor_temperature": 5.0,
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,
        }]
        
        result = validate_simplified_scenario(timeslots)
        
        assert result.valid
        assert len(result.errors) == 0
    
    def test_missing_required_field(self):
        """Missing required field returns invalid."""
        from ml.heating_features import validate_simplified_scenario
        
        next_hour = datetime.now() + timedelta(hours=1)
        timeslots = [{
            "timestamp": next_hour.isoformat(),
            "outdoor_temperature": 5.0,
            # Missing wind_speed, humidity, pressure, target_temperature
        }]
        
        result = validate_simplified_scenario(timeslots)
        
        assert not result.valid
        assert any("Missing required field" in err for err in result.errors)
    
    def test_null_required_field(self):
        """Null required field returns invalid."""
        from ml.heating_features import validate_simplified_scenario
        
        next_hour = datetime.now() + timedelta(hours=1)
        timeslots = [{
            "timestamp": next_hour.isoformat(),
            "outdoor_temperature": None,  # Null value
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,
        }]
        
        result = validate_simplified_scenario(timeslots)
        
        assert not result.valid
        assert any("null" in err.lower() for err in result.errors)
    
    def test_past_timestamp(self):
        """Past timestamp returns invalid."""
        from ml.heating_features import validate_simplified_scenario
        
        past_time = datetime.now() - timedelta(hours=2)
        timeslots = [{
            "timestamp": past_time.isoformat(),
            "outdoor_temperature": 5.0,
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,
        }]
        
        result = validate_simplified_scenario(timeslots)
        
        assert not result.valid
        assert any("future" in err.lower() for err in result.errors)
    
    def test_invalid_timestamp_format(self):
        """Invalid timestamp format returns invalid."""
        from ml.heating_features import validate_simplified_scenario
        
        timeslots = [{
            "timestamp": "not-a-date",
            "outdoor_temperature": 5.0,
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,
        }]
        
        result = validate_simplified_scenario(timeslots)
        
        assert not result.valid
        assert any("Invalid timestamp" in err or "format" in err.lower() for err in result.errors)
    
    def test_invalid_numeric_field(self):
        """Invalid numeric field returns invalid."""
        from ml.heating_features import validate_simplified_scenario
        
        next_hour = datetime.now() + timedelta(hours=1)
        timeslots = [{
            "timestamp": next_hour.isoformat(),
            "outdoor_temperature": "not-a-number",
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,
        }]
        
        result = validate_simplified_scenario(timeslots)
        
        assert not result.valid
        assert any("must be a number" in err for err in result.errors)
    
    def test_multiple_timeslots_validation(self):
        """Multiple timeslots are all validated."""
        from ml.heating_features import validate_simplified_scenario
        
        next_hour = datetime.now() + timedelta(hours=1)
        past_time = datetime.now() - timedelta(hours=1)
        
        timeslots = [
            {  # Valid
                "timestamp": next_hour.isoformat(),
                "outdoor_temperature": 5.0,
                "wind_speed": 3.0,
                "humidity": 75.0,
                "pressure": 1013.0,
                "target_temperature": 20.0,
            },
            {  # Invalid - past timestamp
                "timestamp": past_time.isoformat(),
                "outdoor_temperature": 5.0,
                "wind_speed": 3.0,
                "humidity": 75.0,
                "pressure": 1013.0,
                "target_temperature": 20.0,
            },
        ]
        
        result = validate_simplified_scenario(timeslots)
        
        assert not result.valid
        # Error should reference timeslot 1
        assert any("Timeslot 1" in err for err in result.errors)
    
    def test_optional_indoor_temperature(self):
        """Optional indoor_temperature is validated if present."""
        from ml.heating_features import validate_simplified_scenario
        
        next_hour = datetime.now() + timedelta(hours=1)
        timeslots = [{
            "timestamp": next_hour.isoformat(),
            "outdoor_temperature": 5.0,
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,
            "indoor_temperature": 19.5,  # Optional field
        }]
        
        result = validate_simplified_scenario(timeslots)
        
        assert result.valid
    
    def test_invalid_optional_indoor_temperature(self):
        """Invalid optional indoor_temperature returns invalid."""
        from ml.heating_features import validate_simplified_scenario
        
        next_hour = datetime.now() + timedelta(hours=1)
        timeslots = [{
            "timestamp": next_hour.isoformat(),
            "outdoor_temperature": 5.0,
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,
            "indoor_temperature": "not-a-number",
        }]
        
        result = validate_simplified_scenario(timeslots)
        
        assert not result.valid
        assert any("indoor_temperature" in err and "number" in err for err in result.errors)


class TestConvertSimplifiedToModelFeatures:
    """Tests for convert_simplified_to_model_features function."""
    
    def test_empty_timeslots(self):
        """Empty timeslots returns empty lists."""
        from ml.heating_features import convert_simplified_to_model_features
        
        features, timestamps = convert_simplified_to_model_features([], [])
        
        assert features == []
        assert timestamps == []
    
    def test_basic_conversion(self):
        """Basic conversion maps simplified fields to model fields."""
        from ml.heating_features import convert_simplified_to_model_features
        
        next_hour = datetime.now() + timedelta(hours=1)
        timeslots = [{
            "timestamp": next_hour.isoformat(),
            "outdoor_temperature": 5.0,
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,
            "indoor_temperature": 19.5,
        }]
        
        model_features = ["outdoor_temp", "wind", "humidity", "pressure", "target_temp", "indoor_temp"]
        
        features, timestamps = convert_simplified_to_model_features(
            timeslots, model_features, include_historical_heating=False
        )
        
        assert len(features) == 1
        assert len(timestamps) == 1
        
        # Check field mapping
        assert features[0]["outdoor_temp"] == 5.0
        assert features[0]["wind"] == 3.0
        assert features[0]["humidity"] == 75.0
        assert features[0]["pressure"] == 1013.0
        assert features[0]["target_temp"] == 20.0
        assert features[0]["indoor_temp"] == 19.5
    
    def test_adds_time_features(self):
        """Conversion adds time features from timestamp."""
        from ml.heating_features import convert_simplified_to_model_features
        
        # Create a specific timestamp for predictable results
        ts = datetime(2024, 6, 15, 14, 0, 0)  # Saturday 2pm
        timeslots = [{
            "timestamp": ts.isoformat(),
            "outdoor_temperature": 5.0,
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,
        }]
        
        model_features = ["outdoor_temp", "hour_of_day", "day_of_week", "is_weekend", "is_night"]
        
        features, _ = convert_simplified_to_model_features(
            timeslots, model_features, include_historical_heating=False
        )
        
        assert features[0]["hour_of_day"] == 14
        assert features[0]["day_of_week"] == 5  # Saturday
        assert features[0]["is_weekend"] == 1
        assert features[0]["is_night"] == 0
    
    def test_adds_historical_aggregations(self):
        """Conversion adds historical aggregation features."""
        from ml.heating_features import convert_simplified_to_model_features
        
        next_hour = datetime.now() + timedelta(hours=1)
        timeslots = [
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
                "outdoor_temperature": 4.0,
                "wind_speed": 3.0,
                "humidity": 75.0,
                "pressure": 1013.0,
                "target_temperature": 20.0,
            },
        ]
        
        model_features = ["outdoor_temp", "outdoor_temp_avg_1h", "outdoor_temp_avg_6h"]
        
        features, _ = convert_simplified_to_model_features(
            timeslots, model_features, include_historical_heating=False
        )
        
        # First slot: 1h avg = 5.0 (only one value)
        assert features[0]["outdoor_temp_avg_1h"] == 5.0
        
        # Second slot: 1h avg = 4.0 (current value with window=1)
        assert features[1]["outdoor_temp_avg_1h"] == 4.0
    
    def test_uses_target_temp_for_missing_indoor(self):
        """When indoor_temperature is missing, uses target_temperature."""
        from ml.heating_features import convert_simplified_to_model_features
        
        next_hour = datetime.now() + timedelta(hours=1)
        timeslots = [{
            "timestamp": next_hour.isoformat(),
            "outdoor_temperature": 5.0,
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,
            # No indoor_temperature
        }]
        
        model_features = ["outdoor_temp", "indoor_temp", "target_temp"]
        
        features, _ = convert_simplified_to_model_features(
            timeslots, model_features, include_historical_heating=False
        )
        
        # indoor_temp should be set to target_temp
        assert features[0]["indoor_temp"] == 20.0
    
    def test_computes_heating_degree_hours(self):
        """Conversion computes heating degree hours."""
        from ml.heating_features import convert_simplified_to_model_features
        
        next_hour = datetime.now() + timedelta(hours=1)
        timeslots = [{
            "timestamp": next_hour.isoformat(),
            "outdoor_temperature": 5.0,
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,  # 15 degree difference
        }]
        
        model_features = ["outdoor_temp", "target_temp", "heating_degree_hours_24h"]
        
        features, _ = convert_simplified_to_model_features(
            timeslots, model_features, include_historical_heating=False
        )
        
        # For single slot, heating_degree_hours = (20 - 5) * 1 = 15
        assert features[0]["heating_degree_hours_24h"] == 15.0
    
    def test_parses_timestamp_strings(self):
        """Timestamps are correctly parsed from strings."""
        from ml.heating_features import convert_simplified_to_model_features
        
        ts_str = "2024-06-15T14:00:00"
        timeslots = [{
            "timestamp": ts_str,
            "outdoor_temperature": 5.0,
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,
        }]
        
        _, timestamps = convert_simplified_to_model_features(timeslots, [], False)
        
        assert len(timestamps) == 1
        assert timestamps[0] == datetime(2024, 6, 15, 14, 0, 0)
    
    def test_handles_timezone_in_timestamp(self):
        """Timestamps with timezone info are handled correctly."""
        from ml.heating_features import convert_simplified_to_model_features
        
        ts_str = "2024-06-15T14:00:00Z"  # UTC
        timeslots = [{
            "timestamp": ts_str,
            "outdoor_temperature": 5.0,
            "wind_speed": 3.0,
            "humidity": 75.0,
            "pressure": 1013.0,
            "target_temperature": 20.0,
        }]
        
        _, timestamps = convert_simplified_to_model_features(timeslots, [], False)
        
        assert len(timestamps) == 1
        assert timestamps[0].hour == 14


class TestGetAvailableHistoricalDays:
    """Tests for get_available_historical_days function."""
    
    def test_no_data_returns_empty_list(self, patch_engine):
        """Returns empty list when no data exists."""
        from ml.heating_features import get_available_historical_days
        
        days = get_available_historical_days()
        
        assert days == []
    
    def test_fewer_than_3_days_returns_empty(self, patch_engine):
        """Returns empty list when fewer than 3 days of data exist."""
        from ml.heating_features import get_available_historical_days
        
        start = datetime(2024, 1, 15, 0, 0, 0)
        
        # Create 2 days of data
        with Session(patch_engine) as session:
            for day in range(2):
                for hour in range(24):
                    slot_start = start + timedelta(days=day, hours=hour)
                    session.add(ResampledSample(
                        slot_start=slot_start,
                        category="outdoor_temp",
                        value=5.0,
                        unit="°C",
                    ))
            session.commit()
        
        days = get_available_historical_days()
        
        assert days == []
    
    def test_returns_middle_days_only(self, patch_engine):
        """Returns only middle days, excluding first and last."""
        from ml.heating_features import get_available_historical_days
        
        start = datetime(2024, 1, 15, 0, 0, 0)
        
        # Create 5 days of data
        with Session(patch_engine) as session:
            for day in range(5):
                for hour in range(4):  # Just 4 samples per day for efficiency
                    slot_start = start + timedelta(days=day, hours=hour)
                    session.add(ResampledSample(
                        slot_start=slot_start,
                        category="outdoor_temp",
                        value=5.0,
                        unit="°C",
                    ))
            session.commit()
        
        days = get_available_historical_days()
        
        # Should have 3 days (excluding first and last)
        assert len(days) == 3
        assert "2024-01-16" in days  # Second day
        assert "2024-01-17" in days  # Third day
        assert "2024-01-18" in days  # Fourth day
        assert "2024-01-15" not in days  # First day excluded
        assert "2024-01-19" not in days  # Last day excluded
    
    def test_returns_sorted_dates(self, patch_engine):
        """Returns dates in sorted order."""
        from ml.heating_features import get_available_historical_days
        
        start = datetime(2024, 1, 15, 0, 0, 0)
        
        # Create 4 days of data in random order
        with Session(patch_engine) as session:
            for day in [2, 0, 3, 1]:  # Random order
                slot_start = start + timedelta(days=day)
                session.add(ResampledSample(
                    slot_start=slot_start,
                    category="outdoor_temp",
                    value=5.0,
                    unit="°C",
                ))
            session.commit()
        
        days = get_available_historical_days()
        
        # Should be in sorted order
        assert len(days) == 2
        assert days == ["2024-01-16", "2024-01-17"]


class TestGetHistoricalDayHourlyData:
    """Tests for get_historical_day_hourly_data function."""
    
    def test_invalid_date_format(self, patch_engine):
        """Returns error for invalid date format."""
        from ml.heating_features import get_historical_day_hourly_data
        
        data, error = get_historical_day_hourly_data("not-a-date")
        
        assert data is None
        assert error is not None
        assert "Invalid date format" in error
    
    def test_no_data_for_date(self, patch_engine):
        """Returns error when no data for specified date."""
        from ml.heating_features import get_historical_day_hourly_data
        
        data, error = get_historical_day_hourly_data("2024-01-15")
        
        assert data is None
        assert error is not None
        assert "No data" in error
    
    def test_returns_hourly_data(self, patch_engine):
        """Returns hourly averaged data for a day."""
        from ml.heating_features import get_historical_day_hourly_data
        
        start = datetime(2024, 1, 15, 0, 0, 0)
        
        # Create 24 hours of 5-minute data
        with Session(patch_engine) as session:
            for hour in range(24):
                for minute in range(0, 60, 5):
                    slot_start = start + timedelta(hours=hour, minutes=minute)
                    session.add(ResampledSample(
                        slot_start=slot_start,
                        category="outdoor_temp",
                        value=5.0 + hour * 0.1,
                        unit="°C",
                    ))
                    session.add(ResampledSample(
                        slot_start=slot_start,
                        category="wind",
                        value=3.0,
                        unit="m/s",
                    ))
                    session.add(ResampledSample(
                        slot_start=slot_start,
                        category="humidity",
                        value=75.0,
                        unit="%",
                    ))
                    session.add(ResampledSample(
                        slot_start=slot_start,
                        category="pressure",
                        value=1013.0,
                        unit="hPa",
                    ))
                    session.add(ResampledSample(
                        slot_start=slot_start,
                        category="target_temp",
                        value=20.0,
                        unit="°C",
                    ))
                    session.add(ResampledSample(
                        slot_start=slot_start,
                        category="indoor_temp",
                        value=19.5,
                        unit="°C",
                    ))
                    # hp_kwh_total increases over time
                    session.add(ResampledSample(
                        slot_start=slot_start,
                        category="hp_kwh_total",
                        value=100.0 + hour + minute / 60.0,
                        unit="kWh",
                    ))
            session.commit()
        
        data, error = get_historical_day_hourly_data("2024-01-15")
        
        assert data is not None
        assert error is None
        assert len(data) == 24  # 24 hours
        
        # Check first hour has all expected fields
        first_hour = data[0]
        assert "timestamp" in first_hour
        assert "outdoor_temperature" in first_hour
        assert "wind_speed" in first_hour
        assert "humidity" in first_hour
        assert "pressure" in first_hour
        assert "target_temperature" in first_hour
        assert "indoor_temperature" in first_hour
        assert "actual_heating_kwh" in first_hour
    
    def test_hourly_averaging(self, patch_engine):
        """Data is correctly averaged per hour."""
        from ml.heating_features import get_historical_day_hourly_data
        
        start = datetime(2024, 1, 15, 12, 0, 0)
        
        # Create 12 samples for one hour with known average
        with Session(patch_engine) as session:
            for i in range(12):
                slot_start = start + timedelta(minutes=5 * i)
                # Values from 0 to 11, average should be 5.5
                session.add(ResampledSample(
                    slot_start=slot_start,
                    category="outdoor_temp",
                    value=float(i),
                    unit="°C",
                ))
            session.commit()
        
        data, error = get_historical_day_hourly_data("2024-01-15")
        
        assert data is not None
        assert len(data) == 1
        
        # Average of 0,1,2,3,4,5,6,7,8,9,10,11 = 5.5
        assert data[0]["outdoor_temperature"] == 5.5
    
    def test_hp_kwh_delta_calculation(self, patch_engine):
        """hp_kwh_total is converted to delta (max - min) per hour."""
        from ml.heating_features import get_historical_day_hourly_data
        
        start = datetime(2024, 1, 15, 12, 0, 0)
        
        # Create 12 samples for one hour
        with Session(patch_engine) as session:
            for i in range(12):
                slot_start = start + timedelta(minutes=5 * i)
                # hp_kwh_total increases from 100 to 111
                session.add(ResampledSample(
                    slot_start=slot_start,
                    category="hp_kwh_total",
                    value=100.0 + i,
                    unit="kWh",
                ))
            session.commit()
        
        data, error = get_historical_day_hourly_data("2024-01-15")
        
        assert data is not None
        
        # Delta should be 111 - 100 = 11
        assert data[0]["actual_heating_kwh"] == 11.0
