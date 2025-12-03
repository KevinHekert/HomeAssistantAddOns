"""
Feature statistics calculation module.

This module calculates time-span averages (avg_1h, avg_6h, avg_24h, avg_7d) from
resampled data and stores them in the feature_statistics table.

The calculation process:
1. Reads from resampled_samples table (both raw and derived sensors)
2. Calculates rolling window averages for configured time spans
3. Stores results in feature_statistics table
4. Only calculates for sensors that have the statistic enabled in configuration

This runs AFTER resampling is complete, ensuring all data is present before
calculating averages.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db import FeatureStatistic, ResampledSample
from db.core import engine, init_db_schema
from db.feature_stats import StatType, get_feature_stats_config, sync_stats_config_with_features
from db.sensor_category_config import get_sensor_category_config
from db.virtual_sensors import get_virtual_sensors_config

_Logger = logging.getLogger(__name__)


# Mapping of StatType to time window in minutes
STAT_TYPE_WINDOWS = {
    StatType.AVG_1H: 60,
    StatType.AVG_6H: 360,
    StatType.AVG_24H: 1440,
    StatType.AVG_7D: 10080,
}


@dataclass
class FeatureStatsCalculationResult:
    """Result of feature statistics calculation."""
    stats_calculated: int
    stats_saved: int
    sensors_processed: int
    stat_types_processed: list[str]
    start_time: datetime | None
    end_time: datetime | None


def get_all_sensor_names() -> list[str]:
    """Get all sensor names (raw + virtual) that could have statistics.
    
    Returns sensor names from:
    1. Configured raw sensors (enabled)
    2. Configured virtual sensors (enabled)
    3. Sensors that exist in resampled_samples table
    
    This ensures we can calculate statistics even if sensor configuration isn't set up yet.
    """
    sensor_names = set()
    
    # Get raw sensors from configuration
    try:
        sensor_config = get_sensor_category_config()
        for sensor in sensor_config.get_enabled_sensors():
            sensor_names.add(sensor.category_name)
    except Exception as e:
        _Logger.warning("Could not load sensor category config: %s", e)
    
    # Get virtual sensors from configuration
    try:
        virtual_config = get_virtual_sensors_config()
        for virtual_sensor in virtual_config.get_enabled_sensors():
            sensor_names.add(virtual_sensor.name)
    except Exception as e:
        _Logger.warning("Could not load virtual sensors config: %s", e)
    
    # Also get all unique categories from resampled_samples table
    # This ensures we calculate stats for sensors even if config isn't set up
    try:
        with Session(engine) as session:
            categories = session.query(ResampledSample.category).distinct().all()
            for (category,) in categories:
                sensor_names.add(category)
    except SQLAlchemyError as e:
        _Logger.warning("Could not query resampled samples: %s", e)
    
    return list(sensor_names)


def calculate_rolling_average(
    session: Session,
    sensor_name: str,
    stat_type: StatType,
    slot_start: datetime,
    window_minutes: int,
) -> tuple[float | None, int]:
    """
    Calculate a rolling average for a sensor over a time window.
    
    Args:
        session: Database session
        sensor_name: Name/category of the sensor
        stat_type: Type of statistic to calculate
        slot_start: Start time of the slot for which to calculate the average
        window_minutes: Size of the time window in minutes
        
    Returns:
        Tuple of (average_value, sample_count) or (None, 0) if insufficient data
    """
    # Calculate the window start time (looking backward from slot_start)
    window_start = slot_start - timedelta(minutes=window_minutes)
    
    # Query resampled samples within the window
    result = session.query(
        func.avg(ResampledSample.value),
        func.count(ResampledSample.id)
    ).filter(
        ResampledSample.category == sensor_name,
        ResampledSample.slot_start >= window_start,
        ResampledSample.slot_start < slot_start,
    ).one()
    
    avg_value, sample_count = result
    
    if avg_value is None or sample_count == 0:
        return None, 0
    
    return float(avg_value), int(sample_count)


def calculate_feature_statistics(
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    sync_with_feature_config: bool = True,
) -> FeatureStatsCalculationResult:
    """
    Calculate time-span average statistics for all configured sensors.
    
    This function:
    1. Optionally syncs feature stats config with ML feature config
    2. Gets all enabled sensors (raw + virtual)
    3. For each sensor, checks which statistics are enabled in configuration
    4. Calculates rolling averages from resampled_samples table
    5. Stores results in feature_statistics table
    6. Ensures idempotence by deleting existing stats before inserting
    
    Args:
        start_time: Optional start time for calculation range. If None, uses earliest available.
        end_time: Optional end time for calculation range. If None, uses latest available.
        sync_with_feature_config: If True, syncs feature stats config with ML feature config
            before calculating. This ensures only needed statistics are calculated.
        
    Returns:
        FeatureStatsCalculationResult with statistics about the calculation.
    """
    # Ensure schema exists
    init_db_schema()
    
    # Sync configuration with ML features if requested
    if sync_with_feature_config:
        _Logger.info("Syncing feature stats configuration with ML feature configuration...")
        sync_stats_config_with_features()
    
    # Get configuration
    stats_config = get_feature_stats_config()
    sensor_names = get_all_sensor_names()
    
    if not sensor_names:
        _Logger.warning("No sensors configured for feature statistics calculation")
        return FeatureStatsCalculationResult(
            stats_calculated=0,
            stats_saved=0,
            sensors_processed=0,
            stat_types_processed=[],
            start_time=None,
            end_time=None,
        )
    
    # Determine time range if not provided
    try:
        with Session(engine) as session:
            if start_time is None or end_time is None:
                # Get time range from resampled_samples
                result = session.query(
                    func.min(ResampledSample.slot_start),
                    func.max(ResampledSample.slot_start),
                ).one()
                
                db_start, db_end = result
                
                if db_start is None or db_end is None:
                    _Logger.warning("No resampled samples available for feature statistics calculation")
                    return FeatureStatsCalculationResult(
                        stats_calculated=0,
                        stats_saved=0,
                        sensors_processed=0,
                        stat_types_processed=[],
                        start_time=None,
                        end_time=None,
                    )
                
                if start_time is None:
                    # Use earliest data point as default start
                    # We'll calculate appropriate start times per stat type below
                    start_time = db_start
                    _Logger.info("Using earliest data point as base start_time: %s", start_time)
                
                if end_time is None:
                    end_time = db_end
            
            _Logger.info(
                "Calculating feature statistics from %s to %s for %d sensors",
                start_time,
                end_time,
                len(sensor_names),
            )
            
            stats_calculated = 0
            stats_saved = 0
            stat_types_used = set()
            
            # Process each sensor
            sensors_with_stats = 0
            sensors_skipped_no_config = 0
            sensors_skipped_no_data = 0
            
            for sensor_name in sensor_names:
                # Check if sensor has explicit configuration
                # Don't create defaults automatically
                if sensor_name not in stats_config.sensor_configs:
                    _Logger.debug("No configuration for sensor '%s', skipping", sensor_name)
                    sensors_skipped_no_config += 1
                    continue
                
                sensor_config = stats_config.sensor_configs[sensor_name]
                enabled_stats = sensor_config.enabled_stats
                
                if not enabled_stats:
                    _Logger.debug("No statistics enabled for sensor '%s', skipping", sensor_name)
                    sensors_skipped_no_config += 1
                    continue
                
                # Get all time slots for this sensor in the range
                slots = session.query(ResampledSample.slot_start).filter(
                    ResampledSample.category == sensor_name,
                    ResampledSample.slot_start >= start_time,
                    ResampledSample.slot_start <= end_time,
                ).distinct().order_by(ResampledSample.slot_start).all()
                
                if not slots:
                    _Logger.debug("No resampled data for sensor '%s' in time range", sensor_name)
                    sensors_skipped_no_data += 1
                    continue
                
                sensors_with_stats += 1
                
                # Get unit from first sample
                first_sample = session.query(ResampledSample).filter(
                    ResampledSample.category == sensor_name
                ).first()
                unit = first_sample.unit if first_sample else None
                
                # Calculate statistics for each enabled stat type
                for stat_type in enabled_stats:
                    window_minutes = STAT_TYPE_WINDOWS.get(stat_type)
                    if window_minutes is None:
                        _Logger.warning("Unknown stat type '%s', skipping", stat_type)
                        continue
                    
                    stat_types_used.add(stat_type.value)
                    
                    # Calculate stat-type-specific start time
                    # Each stat type needs enough history for its window
                    stat_start_time = start_time + timedelta(minutes=window_minutes)
                    
                    # Validate this stat type has enough data
                    if stat_start_time > end_time:
                        data_span_minutes = int((end_time - start_time).total_seconds() / 60)
                        _Logger.debug(
                            "Insufficient data for %s on sensor '%s': need %d minutes, have %d minutes. Skipping this stat type.",
                            stat_type.value,
                            sensor_name,
                            window_minutes,
                            data_span_minutes
                        )
                        continue
                    
                    # Get time slots for this sensor in the stat-type-specific range
                    stat_slots = session.query(ResampledSample.slot_start).filter(
                        ResampledSample.category == sensor_name,
                        ResampledSample.slot_start >= stat_start_time,
                        ResampledSample.slot_start <= end_time,
                    ).distinct().order_by(ResampledSample.slot_start).all()
                    
                    if not stat_slots:
                        _Logger.debug(
                            "No slots available for %s on sensor '%s' (need data from %s to %s)",
                            stat_type.value,
                            sensor_name,
                            stat_start_time,
                            end_time
                        )
                        continue
                    
                    # Calculate for each time slot
                    for (slot_start,) in stat_slots:
                        stats_calculated += 1
                        
                        avg_value, sample_count = calculate_rolling_average(
                            session,
                            sensor_name,
                            stat_type,
                            slot_start,
                            window_minutes,
                        )
                        
                        if avg_value is not None:
                            # Delete existing stat (idempotence)
                            session.query(FeatureStatistic).filter(
                                FeatureStatistic.slot_start == slot_start,
                                FeatureStatistic.sensor_name == sensor_name,
                                FeatureStatistic.stat_type == stat_type.value,
                            ).delete()
                            
                            # Insert new stat
                            stat = FeatureStatistic(
                                slot_start=slot_start,
                                sensor_name=sensor_name,
                                stat_type=stat_type.value,
                                value=avg_value,
                                unit=unit,
                                source_sample_count=sample_count,
                            )
                            session.add(stat)
                            stats_saved += 1
                            
                            _Logger.debug(
                                "Calculated %s for %s at %s: %.2f (from %d samples)",
                                stat_type.value,
                                sensor_name,
                                slot_start,
                                avg_value,
                                sample_count,
                            )
            
            # Commit all changes
            session.commit()
            
            _Logger.info(
                "Feature statistics calculation complete: %d calculated, %d saved, %d sensors processed "
                "(%d with stats, %d skipped - no config, %d skipped - no data)",
                stats_calculated,
                stats_saved,
                len(sensor_names),
                sensors_with_stats,
                sensors_skipped_no_config,
                sensors_skipped_no_data,
            )
            
            return FeatureStatsCalculationResult(
                stats_calculated=stats_calculated,
                stats_saved=stats_saved,
                sensors_processed=len(sensor_names),
                stat_types_processed=sorted(stat_types_used),
                start_time=start_time,
                end_time=end_time,
            )
            
    except SQLAlchemyError as e:
        _Logger.error("Error calculating feature statistics: %s", e)
        raise


def get_feature_statistic_value(
    sensor_name: str,
    stat_type: StatType,
    slot_start: datetime,
) -> float | None:
    """
    Get a specific feature statistic value.
    
    Args:
        sensor_name: Name of the sensor
        stat_type: Type of statistic
        slot_start: Time slot
        
    Returns:
        The statistic value or None if not found
    """
    try:
        with Session(engine) as session:
            stat = session.query(FeatureStatistic).filter(
                FeatureStatistic.sensor_name == sensor_name,
                FeatureStatistic.stat_type == stat_type.value,
                FeatureStatistic.slot_start == slot_start,
            ).first()
            
            return stat.value if stat else None
    except SQLAlchemyError as e:
        _Logger.error("Error getting feature statistic: %s", e)
        return None


def flush_feature_statistics() -> int:
    """
    Delete all feature statistics.
    
    This should be called when resampled data changes significantly
    or when recalculating all statistics.
    
    Returns:
        Number of rows deleted.
    """
    try:
        with Session(engine) as session:
            count = session.query(FeatureStatistic).delete()
            session.commit()
            _Logger.info("Flushed feature_statistics table: %d rows deleted", count)
            return count
    except SQLAlchemyError as e:
        _Logger.error("Error flushing feature_statistics table: %s", e)
        raise
