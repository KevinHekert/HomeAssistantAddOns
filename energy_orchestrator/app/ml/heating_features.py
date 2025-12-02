"""
Heating demand feature extraction ETL.

This module builds a feature dataset from resampled sensor samples
for training heating demand prediction models.

Key principles:
- Heat pump outputs (flow/return temperature, HP_POWER_W) are NOT used as input features.
- HP_KWH_TOTAL is used only to compute the target (heating energy demand).
- DHW_ACTIVE is only used to filter out DHW (domestic hot water) slots.
"""

import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from db import ResampledSample
from db.core import engine
from ml.feature_config import get_feature_config

_Logger = logging.getLogger(__name__)


# Required fields for simplified scenario input
SIMPLIFIED_REQUIRED_FIELDS = [
    "timestamp",
    "outdoor_temperature",
    "wind_speed",
    "humidity",
    "pressure",
    "target_temperature",
]

# Optional fields for simplified scenario input
SIMPLIFIED_OPTIONAL_FIELDS = [
    "indoor_temperature",
]


@dataclass
class SimplifiedTimeslot:
    """Simplified scenario input for a single hour timeslot."""
    timestamp: datetime
    outdoor_temperature: float
    wind_speed: float
    humidity: float
    pressure: float
    target_temperature: float
    indoor_temperature: Optional[float] = None


@dataclass
class ScenarioValidationResult:
    """Result of scenario validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Default values for missing model features
DEFAULT_OUTDOOR_TEMP = 5.0  # Default outdoor temperature in °C
DEFAULT_HEATING_KWH = 2.0   # Default heating consumption in kWh
HOURS_PER_DAY = 24          # Hours in a day for heating degree calculation

# Feature categories used as model inputs (exogenous variables only)
INPUT_CATEGORIES = [
    "outdoor_temp",
    "wind",
    "humidity",
    "pressure",
    "indoor_temp",
    "target_temp",
]

# Categories used for target calculation (not as input features)
TARGET_CATEGORIES = [
    "hp_kwh_total",
]

# Categories used for filtering (not as input features)
FILTER_CATEGORIES = [
    "dhw_active",
]

# Prediction horizon in minutes
PREDICTION_HORIZON_MINUTES = 60
SLOTS_PER_HOUR = 12  # 5-minute slots


@dataclass
class TrainingDataRange:
    """First and last values for a training data column."""
    first: Optional[float] = None
    last: Optional[float] = None
    unit: Optional[str] = None


@dataclass
class FeatureDatasetStats:
    """Statistics about the generated feature dataset."""
    total_slots: int
    valid_slots: int
    dropped_missing_features: int
    dropped_missing_target: int
    dropped_insufficient_history: int
    features_used: list[str]
    has_7d_features: bool
    data_start_time: Optional[datetime] = None
    data_end_time: Optional[datetime] = None
    available_history_hours: Optional[float] = None
    # All sensor category ranges (key: category name, value: TrainingDataRange)
    sensor_ranges: dict[str, TrainingDataRange] = field(default_factory=dict)
    # hp_kwh_delta shows the energy consumed during the training period (not raw cumulative values)
    hp_kwh_delta: Optional[float] = None
    # Legacy fields maintained for backward compatibility (use sensor_ranges for new code)
    dhw_temp_range: Optional[TrainingDataRange] = None
    hp_kwh_total_range: Optional[TrainingDataRange] = None


def _load_resampled_data(session: Session) -> pd.DataFrame:
    """
    Load all resampled samples into a DataFrame.
    
    Returns:
        DataFrame with columns: slot_start, category, value, unit
    """
    stmt = select(
        ResampledSample.slot_start,
        ResampledSample.category,
        ResampledSample.value,
        ResampledSample.unit,
    ).order_by(ResampledSample.slot_start)
    
    result = session.execute(stmt).fetchall()
    
    if not result:
        return pd.DataFrame(columns=["slot_start", "category", "value", "unit"])
    
    df = pd.DataFrame(result, columns=["slot_start", "category", "value", "unit"])
    return df


def _pivot_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot data from long to wide format.
    
    Args:
        df: DataFrame with columns slot_start, category, value
        
    Returns:
        DataFrame with slot_start as index and categories as columns
    """
    if df.empty:
        return pd.DataFrame()
    
    pivot = df.pivot(index="slot_start", columns="category", values="value")
    pivot = pivot.sort_index()
    return pivot


def _compute_historical_aggregations(
    pivot_df: pd.DataFrame,
    available_history_hours: float,
) -> pd.DataFrame:
    """
    Compute historical aggregation features.
    
    Args:
        pivot_df: Pivoted DataFrame with slot_start as index
        available_history_hours: Hours of data available
        
    Returns:
        DataFrame with aggregation features added
    """
    df = pivot_df.copy()
    
    # Historical window sizes in number of 5-minute slots
    slots_1h = 12
    slots_6h = 72
    slots_24h = 288
    slots_7d = 2016
    
    # Outdoor temperature history
    if "outdoor_temp" in df.columns:
        df["outdoor_temp_avg_1h"] = df["outdoor_temp"].rolling(
            window=slots_1h, min_periods=1
        ).mean()
        df["outdoor_temp_avg_6h"] = df["outdoor_temp"].rolling(
            window=slots_6h, min_periods=slots_1h
        ).mean()
        df["outdoor_temp_avg_24h"] = df["outdoor_temp"].rolling(
            window=slots_24h, min_periods=slots_6h
        ).mean()
        
        # 7-day average only if we have enough history
        if available_history_hours >= 168:  # 7 days
            df["outdoor_temp_avg_7d"] = df["outdoor_temp"].rolling(
                window=slots_7d, min_periods=slots_24h
            ).mean()
    
    # Indoor temperature history
    if "indoor_temp" in df.columns:
        df["indoor_temp_avg_6h"] = df["indoor_temp"].rolling(
            window=slots_6h, min_periods=slots_1h
        ).mean()
        df["indoor_temp_avg_24h"] = df["indoor_temp"].rolling(
            window=slots_24h, min_periods=slots_6h
        ).mean()
    
    # Target temperature (setpoint) history
    if "target_temp" in df.columns:
        df["target_temp_avg_6h"] = df["target_temp"].rolling(
            window=slots_6h, min_periods=slots_1h
        ).mean()
        df["target_temp_avg_24h"] = df["target_temp"].rolling(
            window=slots_24h, min_periods=slots_6h
        ).mean()
    
    # Heating degree hours
    if "target_temp" in df.columns and "outdoor_temp" in df.columns:
        # Degree difference per slot (5 min = 1/12 hour)
        degree_diff = (df["target_temp"] - df["outdoor_temp"]).clip(lower=0)
        
        # Sum over 24 hours (288 slots * 5 min / 60 = 24 hours)
        # Each slot contributes (degree_diff * 5/60) degree-hours
        df["heating_degree_hours_24h"] = degree_diff.rolling(
            window=slots_24h, min_periods=slots_6h
        ).sum() * (5 / 60)
        
        if available_history_hours >= 168:
            df["heating_degree_hours_7d"] = degree_diff.rolling(
                window=slots_7d, min_periods=slots_24h
            ).sum() * (5 / 60)
    
    # Historical heating kWh (from hp_kwh_total differences)
    if "hp_kwh_total" in df.columns:
        # Compute 5-minute deltas
        kwh_delta = df["hp_kwh_total"].diff()
        
        # Filter out implausible deltas (negative or very large)
        # Assuming max 10 kW heat pump: 10 kW * 5/60 h = 0.833 kWh per 5 min
        kwh_delta = kwh_delta.clip(lower=0, upper=1.0)
        
        # Sum over historical windows (heating only - DHW filtering done later)
        # 1 hour = 12 slots (CORE BASELINE FEATURE)
        df["heating_kwh_last_1h"] = kwh_delta.rolling(
            window=slots_1h, min_periods=1
        ).sum()
        df["heating_kwh_last_6h"] = kwh_delta.rolling(
            window=slots_6h, min_periods=slots_1h
        ).sum()
        df["heating_kwh_last_24h"] = kwh_delta.rolling(
            window=slots_24h, min_periods=slots_6h
        ).sum()
        
        if available_history_hours >= 168:
            df["heating_kwh_last_7d"] = kwh_delta.rolling(
                window=slots_7d, min_periods=slots_24h
            ).sum()
    
    # Derived domain feature: delta between target and indoor temperature (CORE BASELINE)
    if "target_temp" in df.columns and "indoor_temp" in df.columns:
        df["delta_target_indoor"] = df["target_temp"] - df["indoor_temp"]
    
    return df


def _compute_target(
    df: pd.DataFrame,
    horizon_slots: int = SLOTS_PER_HOUR,
) -> pd.DataFrame:
    """
    Compute the target: heating energy demand in kWh over prediction horizon.
    
    Args:
        df: DataFrame with hp_kwh_total column
        horizon_slots: Number of 5-min slots in prediction horizon
        
    Returns:
        DataFrame with target_heating_kwh_1h column added
    """
    if "hp_kwh_total" not in df.columns:
        _Logger.warning("hp_kwh_total not available, cannot compute target")
        df["target_heating_kwh_1h"] = None
        return df
    
    # Compute forward-looking kWh delta
    # For each slot t, compute sum of kWh deltas from t to t+horizon
    kwh_values = df["hp_kwh_total"].values
    n = len(kwh_values)
    target = [None] * n
    
    for i in range(n - horizon_slots):
        start_kwh = kwh_values[i]
        end_kwh = kwh_values[i + horizon_slots]
        
        if pd.notna(start_kwh) and pd.notna(end_kwh):
            delta = end_kwh - start_kwh
            # Filter implausible values
            if 0 <= delta <= 20:  # Max 20 kWh in 1 hour seems reasonable
                target[i] = delta
    
    df["target_heating_kwh_1h"] = target
    
    # Filter out DHW slots if dhw_active is available
    if "dhw_active" in df.columns:
        # Mark slots as DHW if dhw_active > 0.5 (assuming binary 0/1)
        # We need to check the entire horizon window for DHW activity
        dhw_values = df["dhw_active"].values
        
        for i in range(n - horizon_slots):
            if pd.notna(target[i]):
                # Check if any slot in the horizon has DHW active
                horizon_dhw = dhw_values[i:i + horizon_slots]
                if any(pd.notna(v) and v > 0.5 for v in horizon_dhw):
                    target[i] = None  # Exclude DHW slots from training
        
        df["target_heating_kwh_1h"] = target
    
    return df


def _add_time_features(df: pd.DataFrame, use_configured_timezone: bool = True) -> pd.DataFrame:
    """
    Add time-related features.
    
    Args:
        df: DataFrame with DatetimeIndex (assumed UTC)
        use_configured_timezone: If True, convert UTC to configured local timezone
            for hour_of_day. Default is True.
        
    Returns:
        DataFrame with time features added
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Get datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        dt_index = df.index
    else:
        dt_index = pd.DatetimeIndex(df.index)
    
    # Convert to local timezone for hour_of_day if configured
    if use_configured_timezone:
        config = get_feature_config()
        tz = config.get_timezone_info()
        
        try:
            # If index is timezone-naive, assume UTC
            if dt_index.tz is None:
                dt_index_utc = dt_index.tz_localize(timezone.utc)
            else:
                dt_index_utc = dt_index
            
            # Convert to local timezone
            dt_local = dt_index_utc.tz_convert(tz)
            df["hour_of_day"] = dt_local.hour
        except Exception as e:
            _Logger.warning("Error converting timezone, using UTC hours: %s", e)
            df["hour_of_day"] = dt_index.hour
    else:
        df["hour_of_day"] = dt_index.hour
    
    # These features use UTC day of week (consistent with historical data)
    df["day_of_week"] = dt_index.dayofweek
    df["is_weekend"] = (dt_index.dayofweek >= 5).astype(int)
    
    # is_night uses local hour (same as hour_of_day)
    local_hours = df["hour_of_day"]
    df["is_night"] = ((local_hours >= 23) | (local_hours < 7)).astype(int)
    
    return df


def compute_scenario_historical_features(
    scenario_features: list[dict],
    timeslots: Optional[list[datetime]] = None,
) -> list[dict]:
    """
    Compute historical aggregation features from scenario data.
    
    This function derives historical features (1h, 6h, 24h averages) from
    user-provided scenario features. This is useful when making predictions
    for future time periods where historical data doesn't exist yet.
    
    For predictions starting at the next hour, the system uses actual historical
    data. For further predictions, this function can derive historical aggregations
    from the user-provided scenario.
    
    Args:
        scenario_features: List of feature dictionaries with hourly values.
            Must include: outdoor_temp, wind, humidity, pressure, 
            indoor_temp, target_temp
        timeslots: Optional list of timestamps for each slot. If not provided,
            time features (hour_of_day, day_of_week, etc.) must be included
            in scenario_features.
            
    Returns:
        List of feature dictionaries with historical aggregations added.
        
    Example:
        >>> scenario = [
        ...     {"outdoor_temp": 5.0, "target_temp": 20.0, ...},
        ...     {"outdoor_temp": 4.5, "target_temp": 20.0, ...},
        ... ]
        >>> enriched = compute_scenario_historical_features(scenario)
        >>> # Now each slot has outdoor_temp_avg_1h, heating_degree_hours_24h, etc.
    """
    if not scenario_features:
        return []
    
    # Convert to DataFrame for easier computation
    df = pd.DataFrame(scenario_features)
    n_slots = len(df)
    
    # If timeslots provided, use them for time features
    if timeslots:
        dt_index = pd.DatetimeIndex(timeslots)
        df.index = dt_index
        
        # Add time features if not present
        if "hour_of_day" not in df.columns:
            df["hour_of_day"] = dt_index.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = dt_index.dayofweek
        if "is_weekend" not in df.columns:
            df["is_weekend"] = (dt_index.dayofweek >= 5).astype(int)
        if "is_night" not in df.columns:
            df["is_night"] = ((dt_index.hour >= 23) | (dt_index.hour < 7)).astype(int)
    
    # Compute rolling averages from scenario data
    # For hourly predictions (1 slot = 1 hour in user input):
    # - 1h average = current value (or rolling 1)
    # - 6h average = rolling 6 or expanding mean
    # - 24h average = rolling 24 or expanding mean
    
    # Outdoor temperature historical features
    if "outdoor_temp" in df.columns:
        # Use expanding mean for early slots, then rolling
        if "outdoor_temp_avg_1h" not in df.columns:
            df["outdoor_temp_avg_1h"] = df["outdoor_temp"].rolling(
                window=1, min_periods=1
            ).mean()
        if "outdoor_temp_avg_6h" not in df.columns:
            df["outdoor_temp_avg_6h"] = df["outdoor_temp"].rolling(
                window=min(6, n_slots), min_periods=1
            ).mean()
        if "outdoor_temp_avg_24h" not in df.columns:
            df["outdoor_temp_avg_24h"] = df["outdoor_temp"].rolling(
                window=min(24, n_slots), min_periods=1
            ).mean()
    
    # Indoor temperature historical features
    if "indoor_temp" in df.columns:
        if "indoor_temp_avg_6h" not in df.columns:
            df["indoor_temp_avg_6h"] = df["indoor_temp"].rolling(
                window=min(6, n_slots), min_periods=1
            ).mean()
        if "indoor_temp_avg_24h" not in df.columns:
            df["indoor_temp_avg_24h"] = df["indoor_temp"].rolling(
                window=min(24, n_slots), min_periods=1
            ).mean()
    
    # Target temperature historical features
    if "target_temp" in df.columns:
        if "target_temp_avg_6h" not in df.columns:
            df["target_temp_avg_6h"] = df["target_temp"].rolling(
                window=min(6, n_slots), min_periods=1
            ).mean()
        if "target_temp_avg_24h" not in df.columns:
            df["target_temp_avg_24h"] = df["target_temp"].rolling(
                window=min(24, n_slots), min_periods=1
            ).mean()
    
    # Heating degree hours
    if "target_temp" in df.columns and "outdoor_temp" in df.columns:
        degree_diff = (df["target_temp"] - df["outdoor_temp"]).clip(lower=0)
        if "heating_degree_hours_24h" not in df.columns:
            # For hourly data: each hour contributes degree_diff * 1 hour
            df["heating_degree_hours_24h"] = degree_diff.rolling(
                window=min(24, n_slots), min_periods=1
            ).sum()
    
    # Delta between target and indoor temperature (CORE BASELINE FEATURE)
    if "target_temp" in df.columns and "indoor_temp" in df.columns:
        if "delta_target_indoor" not in df.columns:
            df["delta_target_indoor"] = df["target_temp"] - df["indoor_temp"]
    
    # Convert back to list of dicts
    return df.to_dict(orient="records")


def get_actual_vs_predicted_data(
    start_time: datetime,
    end_time: datetime,
    slot_duration_minutes: int = 60,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Retrieve actual historical data for comparison with model predictions.
    
    This function fetches resampled data for a specific time range and aggregates
    it to the requested slot duration. It returns the actual heating kWh values
    that can be compared against model predictions.
    
    Args:
        start_time: Start of the time range (aligned to slot boundary)
        end_time: End of the time range
        slot_duration_minutes: Duration of each prediction slot (default 60 minutes)
        
    Returns:
        Tuple of (DataFrame with actual values, error message if any)
        DataFrame contains: slot_start, outdoor_temp, wind, humidity, pressure,
            indoor_temp, target_temp, actual_heating_kwh, and computed features.
            
    Example:
        >>> from datetime import datetime, timedelta
        >>> start = datetime(2024, 1, 15, 12, 0, 0)
        >>> end = start + timedelta(hours=24)
        >>> df, error = get_actual_vs_predicted_data(start, end)
        >>> if df is not None:
        ...     # Compare df['actual_heating_kwh'] with model predictions
    """
    try:
        with Session(engine) as session:
            # Load resampled data for the time range
            stmt = select(
                ResampledSample.slot_start,
                ResampledSample.category,
                ResampledSample.value,
            ).where(
                ResampledSample.slot_start >= start_time,
                ResampledSample.slot_start < end_time,
            ).order_by(ResampledSample.slot_start)
            
            result = session.execute(stmt).fetchall()
            
            if not result:
                return None, f"No data available for time range {start_time} to {end_time}"
            
            # Convert to DataFrame
            raw_df = pd.DataFrame(result, columns=["slot_start", "category", "value"])
            
            # Pivot to wide format
            pivot_df = raw_df.pivot(
                index="slot_start", 
                columns="category", 
                values="value"
            ).sort_index()
            
            if pivot_df.empty:
                return None, "No data after pivoting"
            
            # Aggregate to requested slot duration (default 1 hour = 12 five-minute slots)
            slots_per_agg = slot_duration_minutes // 5
            
            if slots_per_agg > 1:
                # Group by hour slots and aggregate
                # Floor to nearest slot_duration_minutes boundary
                pivot_df["slot_group"] = (
                    pivot_df.index.to_series()
                    .apply(lambda x: x.replace(
                        minute=(x.minute // slot_duration_minutes) * slot_duration_minutes,
                        second=0, 
                        microsecond=0
                    ))
                )
                
                # For most categories: take the mean
                # For hp_kwh_total: take the max - min (the delta)
                agg_funcs = {}
                for col in pivot_df.columns:
                    if col == "slot_group":
                        continue
                    elif col == "hp_kwh_total":
                        # Use default parameter to capture col value
                        def hp_kwh_agg(x, c=col):
                            return x.max() - x.min() if len(x) > 0 else None
                        agg_funcs[col] = hp_kwh_agg
                    else:
                        agg_funcs[col] = "mean"
                
                hourly_df = pivot_df.groupby("slot_group").agg(agg_funcs)
                hourly_df.index.name = "slot_start"
            else:
                hourly_df = pivot_df.copy()
            
            # Rename hp_kwh_total to actual_heating_kwh
            if "hp_kwh_total" in hourly_df.columns:
                hourly_df = hourly_df.rename(columns={"hp_kwh_total": "actual_heating_kwh"})
            
            # Add time features
            hourly_df = _add_time_features(hourly_df)
            
            # Reset index for output
            hourly_df = hourly_df.reset_index()
            
            return hourly_df, None
            
    except Exception as e:
        _Logger.error("Error fetching actual data: %s", e, exc_info=True)
        return None, str(e)


def validate_prediction_start_time(start_time: datetime) -> tuple[bool, str]:
    """
    Validate that a prediction start time is at the next hour or later.
    
    Predictions should always start at the next or coming hour to ensure
    we have the latest historical data for accurate predictions.
    
    Args:
        start_time: The proposed start time for predictions
        
    Returns:
        Tuple of (is_valid, message)
        
    Example:
        >>> from datetime import datetime, timedelta
        >>> now = datetime.now()
        >>> next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        >>> is_valid, msg = validate_prediction_start_time(next_hour)
        >>> assert is_valid
    """
    now = datetime.now()
    
    # Calculate the next hour boundary
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    
    # Check if start_time is aligned to hour boundary
    if start_time.minute != 0 or start_time.second != 0:
        return False, f"Prediction start time must be aligned to hour boundary. Got: {start_time}"
    
    # Check if start_time is in the past
    if start_time < now:
        return False, f"Prediction start time cannot be in the past. Got: {start_time}, Now: {now}"
    
    # Check if start_time is at least at the next hour
    # Allow some tolerance for current hour if we're very early in it
    if start_time < now.replace(minute=0, second=0, microsecond=0):
        return False, f"Prediction start time must be at least the current hour. Got: {start_time}"
    
    return True, f"Valid prediction start time: {start_time}"


def build_heating_feature_dataset(
    min_samples: int = 100,
) -> tuple[Optional[pd.DataFrame], FeatureDatasetStats]:
    """
    Build the heating demand feature dataset from resampled samples.
    
    This function:
    1. Loads resampled samples from the database
    2. Pivots data to wide format
    3. Computes historical aggregation features
    4. Computes the target (heating energy demand)
    5. Adds time features
    6. Returns a clean dataset ready for model training
    
    Args:
        min_samples: Minimum number of valid samples required
        
    Returns:
        Tuple of (feature_dataframe, stats)
        feature_dataframe is None if insufficient data
    """
    stats = FeatureDatasetStats(
        total_slots=0,
        valid_slots=0,
        dropped_missing_features=0,
        dropped_missing_target=0,
        dropped_insufficient_history=0,
        features_used=[],
        has_7d_features=False,
    )
    
    try:
        with Session(engine) as session:
            # Step 1: Load data
            _Logger.info("Loading resampled samples...")
            raw_df = _load_resampled_data(session)
            
            if raw_df.empty:
                _Logger.warning("No resampled samples found")
                return None, stats
            
            # Step 2: Pivot data
            pivot_df = _pivot_data(raw_df)
            stats.total_slots = len(pivot_df)
            
            if pivot_df.empty:
                _Logger.warning("No data after pivoting")
                return None, stats
            
            _Logger.info(
                "Loaded %d time slots with categories: %s",
                len(pivot_df),
                list(pivot_df.columns),
            )
            
            # Calculate available history and populate stats
            data_start = pivot_df.index.min()
            data_end = pivot_df.index.max()
            time_range = data_end - data_start
            available_hours = time_range.total_seconds() / 3600
            
            stats.data_start_time = data_start.to_pydatetime() if hasattr(data_start, 'to_pydatetime') else data_start
            stats.data_end_time = data_end.to_pydatetime() if hasattr(data_end, 'to_pydatetime') else data_end
            stats.available_history_hours = available_hours
            
            _Logger.info(
                "Available history: %.1f hours (from %s to %s)", 
                available_hours,
                data_start,
                data_end,
            )
            
            # Extract units per category from raw data (use first non-null unit for each category)
            category_units: dict[str, str | None] = {}
            for category in raw_df["category"].unique():
                cat_data = raw_df[raw_df["category"] == category]
                units = cat_data["unit"].dropna()
                if not units.empty:
                    category_units[category] = str(units.iloc[0])
                else:
                    category_units[category] = None
            
            # Capture training data range for all sensor categories
            for category in pivot_df.columns:
                category_values = pivot_df[category].dropna()
                if not category_values.empty:
                    stats.sensor_ranges[category] = TrainingDataRange(
                        first=float(category_values.iloc[0]),
                        last=float(category_values.iloc[-1]),
                        unit=category_units.get(category),
                    )
            
            # For hp_kwh_total, compute the delta (energy consumed during training period)
            if "hp_kwh_total" in pivot_df.columns:
                hp_values = pivot_df["hp_kwh_total"].dropna()
                if not hp_values.empty:
                    stats.hp_kwh_delta = float(hp_values.iloc[-1]) - float(hp_values.iloc[0])
                    # Also set legacy fields for backward compatibility
                    stats.hp_kwh_total_range = TrainingDataRange(
                        first=float(hp_values.iloc[0]),
                        last=float(hp_values.iloc[-1]),
                        unit=category_units.get("hp_kwh_total"),
                    )
            
            # Set legacy dhw_temp_range for backward compatibility
            if "dhw_temp" in pivot_df.columns:
                dhw_values = pivot_df["dhw_temp"].dropna()
                if not dhw_values.empty:
                    stats.dhw_temp_range = TrainingDataRange(
                        first=float(dhw_values.iloc[0]),
                        last=float(dhw_values.iloc[-1]),
                        unit=category_units.get("dhw_temp"),
                    )
            
            # Step 3: Compute historical aggregations
            df = _compute_historical_aggregations(pivot_df, available_hours)
            
            # Step 4: Compute target
            df = _compute_target(df)
            
            # Step 5: Add time features
            df = _add_time_features(df)
            
            # Step 6: Select features for the model using feature configuration
            config = get_feature_config()
            
            # Get active feature names from configuration
            active_feature_names = config.get_active_feature_names()
            
            # Filter to features that are actually available in the data
            available_features = [f for f in active_feature_names if f in df.columns]
            
            # 7-day features are only available if we have enough history
            features_7d = ["outdoor_temp_avg_7d", "heating_degree_hours_7d", "heating_kwh_last_7d"]
            if available_hours >= 168:
                stats.has_7d_features = True
            else:
                # Remove 7d features if not enough history
                available_features = [f for f in available_features if f not in features_7d]
            
            target_col = "target_heating_kwh_1h"
            
            # Filter to rows with valid target
            before_target_filter = len(df)
            df_valid = df[df[target_col].notna()].copy()
            stats.dropped_missing_target = before_target_filter - len(df_valid)
            
            if df_valid.empty:
                _Logger.warning("No valid target values found")
                return None, stats
            
            # Filter to rows with all required features
            before_feature_filter = len(df_valid)
            
            # Check which features have enough non-null values
            final_features = []
            for feat in available_features:
                if feat not in df_valid.columns:
                    _Logger.debug("Feature %s not in data columns, skipping", feat)
                    continue
                null_ratio = df_valid[feat].isna().mean()
                if null_ratio < 0.5:  # Keep features with <50% missing
                    final_features.append(feat)
                else:
                    _Logger.info("Dropping feature %s (%.1f%% missing)", feat, null_ratio * 100)
            
            # Drop rows with any missing values in final features + target
            cols_to_check = final_features + [target_col]
            df_clean = df_valid[cols_to_check].dropna()
            
            stats.dropped_missing_features = before_feature_filter - len(df_clean)
            stats.valid_slots = len(df_clean)
            stats.features_used = final_features
            
            _Logger.info(
                "Dataset stats: total=%d, valid=%d, dropped_target=%d, dropped_features=%d",
                stats.total_slots,
                stats.valid_slots,
                stats.dropped_missing_target,
                stats.dropped_missing_features,
            )
            _Logger.info("Features used: %s", final_features)
            
            if stats.valid_slots < min_samples:
                _Logger.warning(
                    "Insufficient samples: %d < %d required",
                    stats.valid_slots,
                    min_samples,
                )
                return None, stats
            
            return df_clean, stats
            
    except Exception as e:
        _Logger.error("Error building feature dataset: %s", e, exc_info=True)
        return None, stats


def validate_simplified_scenario(
    timeslots: list[dict],
) -> ScenarioValidationResult:
    """
    Validate simplified scenario input for heating demand predictions.
    
    This function validates user-provided scenario timeslots to ensure they
    contain all required fields and have valid values for prediction.
    
    Validation Rules:
    - Scenario must contain at least one timeslot
    - Each timeslot must have all required fields (see SIMPLIFIED_REQUIRED_FIELDS)
    - Timestamps must be ISO 8601 format and in the future
    - All numeric fields must be valid numbers
    - Optional fields (like indoor_temperature) are validated if present
    
    Args:
        timeslots: List of simplified timeslot dictionaries. Each dict should have:
            - timestamp: ISO 8601 datetime string (must be in future)
            - outdoor_temperature: float (°C)
            - wind_speed: float (m/s)
            - humidity: float (%)
            - pressure: float (hPa)
            - target_temperature: float (°C)
            - indoor_temperature: float, optional (°C)
        
    Returns:
        ScenarioValidationResult with:
            - valid: bool indicating if all validations passed
            - errors: list of error messages for each failed validation
            - warnings: list of warning messages (non-blocking issues)
            
    Example:
        >>> from datetime import datetime, timedelta
        >>> next_hour = datetime.now() + timedelta(hours=1)
        >>> slots = [{
        ...     "timestamp": next_hour.isoformat(),
        ...     "outdoor_temperature": 5.0,
        ...     "wind_speed": 3.0,
        ...     "humidity": 75.0,
        ...     "pressure": 1013.0,
        ...     "target_temperature": 20.0,
        ... }]
        >>> result = validate_simplified_scenario(slots)
        >>> assert result.valid
    """
    result = ScenarioValidationResult(valid=True)
    
    if not timeslots:
        result.valid = False
        result.errors.append("Scenario must contain at least one timeslot")
        return result
    
    now = datetime.now()
    
    for idx, slot in enumerate(timeslots):
        slot_errors = []
        
        # Check required fields
        for field in SIMPLIFIED_REQUIRED_FIELDS:
            if field not in slot:
                slot_errors.append(f"Missing required field: {field}")
            elif slot[field] is None:
                slot_errors.append(f"Field '{field}' cannot be null")
        
        if slot_errors:
            result.valid = False
            result.errors.extend([f"Timeslot {idx}: {e}" for e in slot_errors])
            continue
        
        # Parse and validate timestamp
        try:
            ts_value = slot["timestamp"]
            if isinstance(ts_value, str):
                ts = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                # Convert to naive datetime for comparison
                if ts.tzinfo is not None:
                    ts = ts.replace(tzinfo=None)
            elif isinstance(ts_value, datetime):
                ts = ts_value
            else:
                result.valid = False
                result.errors.append(f"Timeslot {idx}: timestamp must be ISO 8601 string or datetime")
                continue
            
            # Validate timestamp is in the future
            if ts <= now:
                result.valid = False
                result.errors.append(
                    f"Timeslot {idx}: timestamp must be in the future. "
                    f"Got: {ts.isoformat()}, Now: {now.isoformat()}"
                )
        except ValueError as e:
            result.valid = False
            result.errors.append(f"Timeslot {idx}: Invalid timestamp format: {e}")
            continue
        
        # Validate numeric fields
        numeric_fields = [
            "outdoor_temperature",
            "wind_speed",
            "humidity",
            "pressure",
            "target_temperature",
        ]
        for field in numeric_fields:
            value = slot.get(field)
            if value is not None:
                try:
                    float(value)
                except (ValueError, TypeError):
                    result.valid = False
                    result.errors.append(
                        f"Timeslot {idx}: Field '{field}' must be a number"
                    )
        
        # Validate optional indoor_temperature if present
        indoor_temp = slot.get("indoor_temperature")
        if indoor_temp is not None:
            try:
                float(indoor_temp)
            except (ValueError, TypeError):
                result.valid = False
                result.errors.append(
                    f"Timeslot {idx}: Field 'indoor_temperature' must be a number"
                )
    
    return result


def get_historical_heating_kwh(hours: int = 24) -> dict[str, float]:
    """
    Get historical heating kWh values from the database.
    
    Args:
        hours: Number of hours of history to retrieve
        
    Returns:
        Dictionary with heating_kwh_last_6h and heating_kwh_last_24h
    """
    result = {
        "heating_kwh_last_6h": 0.0,
        "heating_kwh_last_24h": 0.0,
    }
    
    try:
        with Session(engine) as session:
            now = datetime.now()
            start_6h = now - timedelta(hours=6)
            start_24h = now - timedelta(hours=24)
            
            # Get hp_kwh_total values for the time range
            stmt = select(
                ResampledSample.slot_start,
                ResampledSample.value,
            ).where(
                ResampledSample.category == "hp_kwh_total",
                ResampledSample.slot_start >= start_24h,
            ).order_by(ResampledSample.slot_start)
            
            rows = session.execute(stmt).fetchall()
            
            if len(rows) < 2:
                return result
            
            # Compute kWh deltas
            values = [(r[0], r[1]) for r in rows]
            
            # Last 6 hours
            values_6h = [v for v in values if v[0] >= start_6h]
            if len(values_6h) >= 2:
                result["heating_kwh_last_6h"] = values_6h[-1][1] - values_6h[0][1]
            
            # Last 24 hours
            if len(values) >= 2:
                result["heating_kwh_last_24h"] = values[-1][1] - values[0][1]
            
            # Clamp to reasonable values (0-100 kWh)
            result["heating_kwh_last_6h"] = max(0.0, min(100.0, result["heating_kwh_last_6h"]))
            result["heating_kwh_last_24h"] = max(0.0, min(100.0, result["heating_kwh_last_24h"]))
            
    except Exception as e:
        _Logger.warning("Failed to get historical heating kWh: %s", e)
    
    return result


def convert_simplified_to_model_features(
    timeslots: list[dict],
    model_feature_names: list[str],
    include_historical_heating: bool = True,
) -> tuple[list[dict], list[datetime]]:
    """
    Convert simplified scenario input to model-ready features.
    
    This function takes user-friendly inputs (weather forecast, setpoints)
    and converts them to the full feature set required by the model.
    
    The following features are derived internally:
    - Time features: hour_of_day, day_of_week, is_weekend, is_night
    - Historical aggregations: outdoor_temp_avg_*, indoor_temp_avg_*, 
      target_temp_avg_*, heating_degree_hours_24h
    - Historical heating: heating_kwh_last_* (from DB or defaults)
    
    Args:
        timeslots: List of simplified timeslot dictionaries with:
            - timestamp: ISO 8601 datetime string
            - outdoor_temperature: Predicted outdoor temp
            - wind_speed: Predicted wind speed
            - humidity: Predicted humidity
            - pressure: Predicted pressure
            - target_temperature: Planned setpoint
            - indoor_temperature: (optional) Expected indoor temp
        model_feature_names: List of features the model expects
        include_historical_heating: Whether to fetch historical heating kWh
        
    Returns:
        Tuple of (model_features list, parsed_timestamps list)
        
    Example:
        >>> slots = [
        ...     {
        ...         "timestamp": "2024-01-15T14:00:00",
        ...         "outdoor_temperature": 5.0,
        ...         "wind_speed": 3.0,
        ...         "humidity": 75.0,
        ...         "pressure": 1013.0,
        ...         "target_temperature": 20.0,
        ...     }
        ... ]
        >>> features, timestamps = convert_simplified_to_model_features(
        ...     slots, model.feature_names
        ... )
    """
    if not timeslots:
        return [], []
    
    # Parse timestamps
    parsed_timestamps = []
    for slot in timeslots:
        ts_value = slot["timestamp"]
        if isinstance(ts_value, str):
            ts = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
        else:
            ts = ts_value
        parsed_timestamps.append(ts)
    
    # Convert simplified fields to model field names
    # Simplified name -> model name mapping
    field_mapping = {
        "outdoor_temperature": "outdoor_temp",
        "wind_speed": "wind",
        "humidity": "humidity",
        "pressure": "pressure",
        "target_temperature": "target_temp",
        "indoor_temperature": "indoor_temp",
    }
    
    # Build base features
    base_features = []
    for slot in timeslots:
        features = {}
        for simple_name, model_name in field_mapping.items():
            if simple_name in slot and slot[simple_name] is not None:
                features[model_name] = float(slot[simple_name])
        base_features.append(features)
    
    # Add default indoor_temp if not provided (use target_temp as estimate)
    for i, features in enumerate(base_features):
        if "indoor_temp" not in features and "target_temp" in features:
            # Use target temp as indoor temp estimate
            features["indoor_temp"] = features["target_temp"]
    
    # Get historical heating kWh from database
    historical_heating = {}
    if include_historical_heating:
        historical_heating = get_historical_heating_kwh()
    
    # Add historical heating to first slot, then rolling sum for subsequent
    for i, features in enumerate(base_features):
        if "heating_kwh_last_6h" in model_feature_names:
            features["heating_kwh_last_6h"] = historical_heating.get("heating_kwh_last_6h", DEFAULT_HEATING_KWH)
        if "heating_kwh_last_24h" in model_feature_names:
            features["heating_kwh_last_24h"] = historical_heating.get("heating_kwh_last_24h", DEFAULT_HEATING_KWH * 5)
    
    # Enrich with historical aggregations and time features
    enriched_features = compute_scenario_historical_features(
        base_features,
        timeslots=parsed_timestamps,
    )
    
    # Ensure all model features are present
    final_features = []
    for features in enriched_features:
        final_slot = {}
        for feat in model_feature_names:
            if feat in features:
                final_slot[feat] = features[feat]
            else:
                # Provide default values for missing features
                if "avg" in feat:
                    # Use corresponding base value
                    base_name = feat.split("_avg_")[0]
                    final_slot[feat] = features.get(base_name, DEFAULT_OUTDOOR_TEMP)
                elif feat.startswith("heating_kwh"):
                    final_slot[feat] = historical_heating.get(feat, DEFAULT_HEATING_KWH)
                elif feat == "heating_degree_hours_24h":
                    # Compute from target and outdoor if available
                    target = features.get("target_temp", 20.0)
                    outdoor = features.get("outdoor_temp", DEFAULT_OUTDOOR_TEMP)
                    final_slot[feat] = max(0, target - outdoor) * HOURS_PER_DAY
                else:
                    final_slot[feat] = 0.0
        final_features.append(final_slot)
    
    return final_features, parsed_timestamps


def get_available_historical_days() -> list[str]:
    """
    Get list of available days from resampled samples, excluding first and last day.
    
    This function queries the resampled_samples table to find all unique dates
    with data, then returns only the middle days (excluding the first and last day)
    to ensure complete data is available.
    
    Returns:
        List of date strings in YYYY-MM-DD format, sorted chronologically.
        Returns empty list if fewer than 3 days of data exist.
        
    Example:
        >>> days = get_available_historical_days()
        >>> print(days)  # e.g., ['2024-01-02', '2024-01-03', '2024-01-04']
    """
    try:
        with Session(engine) as session:
            # Get distinct dates from resampled samples
            stmt = select(
                ResampledSample.slot_start,
            ).distinct().order_by(ResampledSample.slot_start)
            
            result = session.execute(stmt).fetchall()
            
            if not result:
                return []
            
            # Extract unique dates
            dates = sorted(set(
                row[0].date() for row in result
            ))
            
            # Need at least 3 days to return middle days
            if len(dates) < 3:
                _Logger.info(
                    "Fewer than 3 days of data (%d), cannot return historical days",
                    len(dates),
                )
                return []
            
            # Exclude first and last day
            middle_dates = dates[1:-1]
            
            return [d.isoformat() for d in middle_dates]
            
    except Exception as e:
        _Logger.error("Error getting available historical days: %s", e, exc_info=True)
        return []


def get_historical_day_hourly_data(
    date_str: str,
) -> tuple[Optional[list[dict]], Optional[str]]:
    """
    Get hourly averaged data for a specific historical day.
    
    This function retrieves 5-minute resampled data for the specified day
    and aggregates it to hourly averages for use as scenario input.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        
    Returns:
        Tuple of (hourly_data list, error message if any)
        
        hourly_data is a list of 24 dictionaries, one per hour, containing:
        - timestamp: ISO 8601 timestamp for that hour
        - outdoor_temperature: Hourly average outdoor temp
        - wind_speed: Hourly average wind speed
        - humidity: Hourly average humidity
        - pressure: Hourly average pressure
        - target_temperature: Hourly average target temp
        - indoor_temperature: Hourly average indoor temp (if available)
        - actual_heating_kwh: Actual heating kWh for that hour (from hp_kwh_total delta)
        
    Example:
        >>> data, error = get_historical_day_hourly_data("2024-01-15")
        >>> if data:
        ...     print(len(data))  # 24 hours
        ...     print(data[0]["outdoor_temperature"])
    """
    try:
        # Parse the date
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return None, f"Invalid date format: {date_str}. Use YYYY-MM-DD."
        
        # Define time range for the entire day
        start_time = datetime.combine(target_date, datetime.min.time())
        end_time = start_time + timedelta(days=1)
        
        with Session(engine) as session:
            # Load resampled data for the day
            stmt = select(
                ResampledSample.slot_start,
                ResampledSample.category,
                ResampledSample.value,
            ).where(
                ResampledSample.slot_start >= start_time,
                ResampledSample.slot_start < end_time,
            ).order_by(ResampledSample.slot_start)
            
            result = session.execute(stmt).fetchall()
            
            if not result:
                return None, f"No data available for date {date_str}"
            
            # Convert to DataFrame
            raw_df = pd.DataFrame(result, columns=["slot_start", "category", "value"])
            
            # Pivot to wide format
            pivot_df = raw_df.pivot(
                index="slot_start",
                columns="category",
                values="value",
            ).sort_index()
            
            if pivot_df.empty:
                return None, "No data after pivoting"
            
            # Add hour column for grouping
            pivot_df["hour"] = pivot_df.index.to_series().apply(
                lambda x: x.replace(minute=0, second=0, microsecond=0)
            )
            
            # Aggregate by hour
            # For most columns: mean
            # For hp_kwh_total: max - min (delta for the hour)
            def _hp_kwh_agg(x):
                """Compute kWh delta for the hour (max - min)."""
                if len(x) > 0 and not x.isna().all():
                    return x.max() - x.min()
                return None
            
            agg_funcs = {}
            for col in pivot_df.columns:
                if col == "hour":
                    continue
                elif col == "hp_kwh_total":
                    agg_funcs[col] = _hp_kwh_agg
                else:
                    agg_funcs[col] = "mean"
            
            hourly_df = pivot_df.groupby("hour").agg(agg_funcs)
            
            # Map category names to simplified names
            category_to_simplified = {
                "outdoor_temp": "outdoor_temperature",
                "wind": "wind_speed",
                "humidity": "humidity",
                "pressure": "pressure",
                "target_temp": "target_temperature",
                "indoor_temp": "indoor_temperature",
                "hp_kwh_total": "actual_heating_kwh",
            }
            
            # Build result list
            hourly_data = []
            for hour_ts, row in hourly_df.iterrows():
                hour_dict = {
                    "timestamp": hour_ts.isoformat(),
                }
                
                for cat, simplified in category_to_simplified.items():
                    if cat in row and pd.notna(row[cat]):
                        hour_dict[simplified] = round(float(row[cat]), 4)
                
                hourly_data.append(hour_dict)
            
            # Sort by timestamp
            hourly_data.sort(key=lambda x: x["timestamp"])
            
            return hourly_data, None
            
    except Exception as e:
        _Logger.error(
            "Error getting historical day data for %s: %s",
            date_str,
            e,
            exc_info=True,
        )
        return None, str(e)
