"""
Resampling logic for aggregating raw sensor samples into configurable time slots.

This module provides functionality to:
1. Map logical categories to Home Assistant entities
2. Compute time-weighted averages for sensor data
3. Resample raw samples into uniform time slots (configurable, default 5 minutes)
4. Calculate virtual (derived) sensor values from raw sensor data
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db import ResampledSample, Sample, SensorMapping
from db.core import engine, init_db_schema
from db.virtual_sensors import get_virtual_sensors_config

_Logger = logging.getLogger(__name__)

# Valid sample rates that divide evenly into 60 minutes
VALID_SAMPLE_RATES = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]

# Default step size for resampling (used for backward compatibility)
RESAMPLE_STEP = timedelta(minutes=5)

# Default sample rate in minutes
DEFAULT_SAMPLE_RATE = 5

# Configuration file path for persistent sample rate storage
# In Home Assistant add-ons, /data is the persistent data directory
CONFIG_FILE_PATH = Path(os.environ.get("DATA_DIR", "/data")) / "resample_config.json"


def _load_sample_rate_config() -> int:
    """Load sample rate from persistent configuration file.
    
    Returns:
        Sample rate in minutes, defaults to 5 if not configured or invalid.
    """
    try:
        if CONFIG_FILE_PATH.exists():
            with open(CONFIG_FILE_PATH, "r") as f:
                config = json.load(f)
                rate = config.get("sample_rate_minutes", DEFAULT_SAMPLE_RATE)
                if isinstance(rate, int) and rate in VALID_SAMPLE_RATES:
                    return rate
                _Logger.warning(
                    "Invalid sample rate %s in config. Valid rates are %s. Using default %d minutes",
                    rate,
                    VALID_SAMPLE_RATES,
                    DEFAULT_SAMPLE_RATE,
                )
    except (json.JSONDecodeError, OSError) as e:
        _Logger.warning("Error loading sample rate config: %s. Using default %d minutes", e, DEFAULT_SAMPLE_RATE)
    return DEFAULT_SAMPLE_RATE


def _save_sample_rate_config(rate: int) -> bool:
    """Save sample rate to persistent configuration file.
    
    Args:
        rate: Sample rate in minutes. Must be one of VALID_SAMPLE_RATES.
        
    Returns:
        True if saved successfully, False otherwise.
    """
    if rate not in VALID_SAMPLE_RATES:
        _Logger.error("Cannot save invalid sample rate %d. Valid rates are %s", rate, VALID_SAMPLE_RATES)
        return False
    
    try:
        # Ensure parent directory exists
        CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing config or create new
        config = {}
        if CONFIG_FILE_PATH.exists():
            try:
                with open(CONFIG_FILE_PATH, "r") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        
        # Update sample rate
        config["sample_rate_minutes"] = rate
        
        # Save config
        with open(CONFIG_FILE_PATH, "w") as f:
            json.dump(config, f, indent=2)
        
        _Logger.info("Sample rate saved to config: %d minutes", rate)
        return True
    except OSError as e:
        _Logger.error("Error saving sample rate config: %s", e)
        return False


def get_sample_rate_minutes() -> int:
    """Get the sample rate in minutes from persistent configuration.
    
    Valid sample rates are divisors of 60: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60.
    This ensures that time slots align properly at hour boundaries.
    
    Returns:
        Sample rate in minutes, defaults to 5 if not configured or invalid.
    """
    return _load_sample_rate_config()


def set_sample_rate_minutes(rate: int) -> bool:
    """Set the sample rate in minutes and persist to configuration.
    
    Args:
        rate: Sample rate in minutes. Must be one of VALID_SAMPLE_RATES.
        
    Returns:
        True if saved successfully, False if rate is invalid or save failed.
    """
    if rate not in VALID_SAMPLE_RATES:
        _Logger.warning(
            "Invalid sample rate %d. Valid rates are %s.",
            rate,
            VALID_SAMPLE_RATES,
        )
        return False
    return _save_sample_rate_config(rate)


@dataclass
class ResampleStats:
    """Statistics about the resampling operation."""
    slots_processed: int
    slots_saved: int
    slots_skipped: int
    categories: list[str]
    start_time: datetime | None
    end_time: datetime | None
    sample_rate_minutes: int = 5
    table_flushed: bool = False


def flush_resampled_samples() -> int:
    """
    Delete all records from the resampled_samples table.
    
    This should be called before resampling when the sample rate changes,
    because existing resampled data computed with a different interval
    becomes invalid.
    
    Returns:
        Number of rows deleted.
    """
    try:
        with Session(engine) as session:
            count = session.query(ResampledSample).delete()
            session.commit()
            _Logger.info("Flushed resampled_samples table: %d rows deleted", count)
            return count
    except SQLAlchemyError as e:
        _Logger.error("Error flushing resampled_samples table: %s", e)
        raise


def get_latest_resampled_slot_start() -> datetime | None:
    """
    Get the latest slot_start datetime from the resampled_samples table.
    
    Returns:
        The maximum slot_start datetime, or None if the table is empty.
    """
    try:
        with Session(engine) as session:
            result = session.query(func.max(ResampledSample.slot_start)).scalar()
            return result
    except SQLAlchemyError as e:
        _Logger.error("Error getting latest resampled slot start: %s", e)
        return None


def get_primary_entities_by_category() -> dict[str, str]:
    """
    Determine the primary entity_id per category from sensor_mappings.

    Only considers rows with is_active = TRUE.
    Sorts by: category ASC, priority ASC, id ASC.
    For each category, picks the first row as the primary entity.

    Returns:
        Dictionary mapping category names to their primary entity_id.
        Empty dict if no active mappings exist.
    """
    try:
        with Session(engine) as session:
            mappings = (
                session.query(SensorMapping)
                .filter(SensorMapping.is_active == True)  # noqa: E712
                .order_by(
                    SensorMapping.category.asc(),
                    SensorMapping.priority.asc(),
                    SensorMapping.id.asc(),
                )
                .all()
            )

            # Pick the first entity for each category
            category_to_entity: dict[str, str] = {}
            for mapping in mappings:
                if mapping.category not in category_to_entity:
                    category_to_entity[mapping.category] = mapping.entity_id

            return category_to_entity

    except SQLAlchemyError as e:
        _Logger.error("Error fetching primary entities by category: %s", e)
        return {}


def get_global_range_for_all_categories() -> tuple[datetime | None, datetime | None, dict[str, str]]:
    """
    Compute the global time range where all primary entities have data.

    For each primary entity:
    - Compute min_ts and max_ts from the samples table.

    Global range:
    - global_start = max(min_ts over all primary entities)
    - global_end = min(max_ts over all primary entities)

    Returns:
        Tuple of (global_start, global_end, category_to_entity).
        Returns (None, None, category_to_entity) if any entity has no data.
    """
    category_to_entity = get_primary_entities_by_category()

    if not category_to_entity:
        _Logger.warning("No active sensor mappings found.")
        return None, None, category_to_entity

    global_start: datetime | None = None
    global_end: datetime | None = None

    try:
        with Session(engine) as session:
            for category, entity_id in category_to_entity.items():
                result = session.query(
                    func.min(Sample.timestamp),
                    func.max(Sample.timestamp),
                ).filter(Sample.entity_id == entity_id).one()

                min_ts, max_ts = result

                if min_ts is None or max_ts is None:
                    _Logger.warning(
                        "Category '%s' (entity '%s') has no data in samples table.",
                        category,
                        entity_id,
                    )
                    return None, None, category_to_entity

                # Update global range
                if global_start is None or min_ts > global_start:
                    global_start = min_ts
                if global_end is None or max_ts < global_end:
                    global_end = max_ts

        return global_start, global_end, category_to_entity

    except SQLAlchemyError as e:
        _Logger.error("Error computing global range for categories: %s", e)
        return None, None, category_to_entity


def compute_time_weighted_avg(
    session: Session,
    entity_id: str,
    window_start: datetime,
    window_end: datetime,
) -> tuple[float | None, str | None]:
    """
    Compute a time-weighted average for an entity within a time window.

    Uses last-known-value (piecewise constant / zero-order hold) behavior:
    - Between two sample timestamps, value is the last known sample value.

    Args:
        session: SQLAlchemy session to use for queries.
        entity_id: The entity to compute the average for.
        window_start: Start of the time window (inclusive).
        window_end: End of the time window (exclusive).

    Returns:
        Tuple of (average_value, unit) or (None, None) if no data available.
    """
    try:
        # Find the last sample before window_start
        prev_sample = (
            session.query(Sample)
            .filter(
                Sample.entity_id == entity_id,
                Sample.timestamp < window_start,
            )
            .order_by(Sample.timestamp.desc())
            .first()
        )

        # Load all samples within [window_start, window_end)
        window_samples = (
            session.query(Sample)
            .filter(
                Sample.entity_id == entity_id,
                Sample.timestamp >= window_start,
                Sample.timestamp < window_end,
            )
            .order_by(Sample.timestamp.asc())
            .all()
        )

        # If no previous sample and no samples in window, return None
        if prev_sample is None and not window_samples:
            return None, None

        # Initialize tracking variables
        current_time = window_start
        current_val: float
        current_unit: str | None

        if prev_sample is not None:
            current_val = prev_sample.value
            current_unit = prev_sample.unit
        else:
            # Use first sample in window
            current_val = window_samples[0].value
            current_unit = window_samples[0].unit

        accum = 0.0
        total = 0.0

        # Process each sample in the window
        for sample in window_samples:
            next_time = sample.timestamp
            dt = (next_time - current_time).total_seconds()

            if dt > 0:
                accum += current_val * dt
                total += dt

            current_time = next_time
            current_val = sample.value
            current_unit = sample.unit

        # Process remaining time until window_end
        dt_final = (window_end - current_time).total_seconds()
        if dt_final > 0:
            accum += current_val * dt_final
            total += dt_final

        if total == 0:
            return None, None

        avg = accum / total
        return avg, current_unit

    except SQLAlchemyError as e:
        _Logger.error(
            "Error computing time-weighted average for %s [%s, %s): %s",
            entity_id,
            window_start,
            window_end,
            e,
        )
        return None, None


def _align_to_5min_boundary(dt: datetime) -> datetime:
    """
    Align a datetime downwards to the nearest 5-minute boundary.

    Strips seconds/microseconds and floors minutes to a multiple of 5.
    
    Note: This function uses a fixed 5-minute boundary for backward compatibility.
    For configurable sample rates, use _align_to_boundary().
    """
    aligned_minutes = (dt.minute // 5) * 5
    return dt.replace(minute=aligned_minutes, second=0, microsecond=0)


def _align_to_boundary(dt: datetime, sample_rate_minutes: int) -> datetime:
    """
    Align a datetime downwards to the nearest boundary based on sample rate.

    Strips seconds/microseconds and floors minutes to a multiple of sample_rate_minutes.
    
    Args:
        dt: The datetime to align.
        sample_rate_minutes: The sample rate in minutes (e.g., 5, 10, 15, 30, 60).
        
    Returns:
        Aligned datetime with seconds/microseconds stripped and minutes floored.
    """
    if sample_rate_minutes <= 0:
        sample_rate_minutes = 5
    
    aligned_minutes = (dt.minute // sample_rate_minutes) * sample_rate_minutes
    return dt.replace(minute=aligned_minutes, second=0, microsecond=0)


def resample_all_categories(sample_rate_minutes: int | None = None, flush: bool = False) -> ResampleStats:
    """
    Resample raw sensor samples into time slots for all configured categories.

    This function:
    1. Ensures DB schema exists (creates tables if missing).
    2. Optionally flushes the resampled_samples table (for sample rate changes).
    3. Fetches category-to-entity mappings.
    4. Computes the global time range where all categories have data.
    5. Iterates over time slots and computes time-weighted averages for raw sensors.
    6. Marks raw sensor samples as is_derived=False (sampled from raw sensor data).
    7. Calculates virtual (derived) sensor values from resampled raw sensor data.
    8. Marks virtual sensor samples as is_derived=True (calculated after resampling).
    9. Only writes complete slots (all categories have values).
    10. Ensures idempotence by deleting existing rows before inserting.
    
    **Important**: Raw sensors are sampled first, then virtual sensors are calculated
    from the resampled raw data. This ensures proper data lineage and allows
    distinguishing between raw measurements and derived calculations.
    
    Virtual sensors are only calculated if both source sensors have values in that slot.
    Enabled virtual sensors are loaded from the configuration file.
    
    Args:
        sample_rate_minutes: Optional sample rate in minutes. If None, uses
            the configured SAMPLE_RATE_MINUTES environment variable (default 5).
        flush: If True, flush (delete all) existing resampled data before
            resampling. This should be used when the sample rate changes.

    Returns:
        ResampleStats with statistics about the resampling operation.
    """
    # Use provided sample rate or get from environment
    if sample_rate_minutes is None:
        sample_rate_minutes = get_sample_rate_minutes()
    
    resample_step = timedelta(minutes=sample_rate_minutes)
    
    # Step 1: Ensure DB schema exists
    init_db_schema()
    
    # Step 1.5: Flush existing resampled data if requested
    table_flushed = False
    if flush:
        flush_resampled_samples()
        table_flushed = True

    # Step 2: Fetch mappings
    category_to_entity = get_primary_entities_by_category()

    if not category_to_entity:
        _Logger.warning(
            "No active sensor mappings configured, skipping resample."
        )
        return ResampleStats(
            slots_processed=0,
            slots_saved=0,
            slots_skipped=0,
            categories=[],
            start_time=None,
            end_time=None,
            sample_rate_minutes=sample_rate_minutes,
            table_flushed=table_flushed,
        )

    # Step 3: Get global time range
    global_start, global_end, category_to_entity = get_global_range_for_all_categories()

    if global_start is None or global_end is None:
        _Logger.warning(
            "No global time range available for all categories, skipping resample."
        )
        return ResampleStats(
            slots_processed=0,
            slots_saved=0,
            slots_skipped=0,
            categories=list(category_to_entity.keys()),
            start_time=None,
            end_time=None,
            sample_rate_minutes=sample_rate_minutes,
            table_flushed=table_flushed,
        )

    # Step 4: Align global_start to boundary based on sample rate
    aligned_start = _align_to_boundary(global_start, sample_rate_minutes)
    
    # Step 4.5: For incremental resampling (when flush=False), start from
    # the latest resampled slot minus 2 * sample_rate_minutes.
    # This ensures we re-process recent slots that may have incomplete data.
    effective_start = aligned_start
    if not flush:
        latest_resampled = get_latest_resampled_slot_start()
        if latest_resampled is not None:
            # Go back 2 * sample_rate_minutes from the latest resampled slot
            incremental_start = latest_resampled - timedelta(minutes=2 * sample_rate_minutes)
            # Align to boundary
            incremental_start = _align_to_boundary(incremental_start, sample_rate_minutes)
            # Use incremental start only if it's after the global aligned start
            if incremental_start > aligned_start:
                effective_start = incremental_start
                _Logger.info(
                    "Incremental resample: starting from %s (latest resampled: %s)",
                    effective_start,
                    latest_resampled,
                )

    _Logger.info(
        "Starting resample: effective_start=%s, global_end=%s, categories=%s, sample_rate=%dm",
        effective_start,
        global_end,
        list(category_to_entity.keys()),
        sample_rate_minutes,
    )

    # Track statistics
    slots_processed = 0
    slots_saved = 0
    slots_skipped = 0

    # Step 4.8: Load enabled virtual sensors
    virtual_sensors_config = get_virtual_sensors_config()
    enabled_virtual_sensors = virtual_sensors_config.get_enabled_sensors()
    
    _Logger.info(
        "Found %d enabled virtual sensors for resampling",
        len(enabled_virtual_sensors),
    )

    # Step 5: Iterate over slots
    try:
        with Session(engine) as session:
            slot_start = effective_start

            while slot_start < global_end:
                slot_end = slot_start + resample_step
                slots_processed += 1

                # Compute values for all categories
                slot_values: dict[str, tuple[float, str | None]] = {}
                slot_complete = True

                for category, entity_id in category_to_entity.items():
                    avg, unit = compute_time_weighted_avg(
                        session, entity_id, slot_start, slot_end
                    )

                    if avg is None:
                        _Logger.debug(
                            "Incomplete slot %s: category '%s' has no value.",
                            slot_start,
                            category,
                        )
                        slot_complete = False
                        break

                    slot_values[category] = (avg, unit)

                # Only write if all categories have values
                if slot_complete:
                    # Delete existing rows for this slot (idempotence)
                    session.query(ResampledSample).filter(
                        ResampledSample.slot_start == slot_start
                    ).delete()

                    # Step 5.1: Insert raw sensor samples first (is_derived=False)
                    # These are direct time-weighted averages from raw sensor data
                    for category, (avg, unit) in slot_values.items():
                        resampled = ResampledSample(
                            slot_start=slot_start,
                            category=category,
                            value=avg,
                            unit=unit,
                            is_derived=False,  # Raw sensor data
                        )
                        session.add(resampled)
                    
                    # Step 5.2: Calculate and insert virtual sensor values (is_derived=True)
                    # These are calculated from the resampled raw sensor data above
                    for virtual_sensor in enabled_virtual_sensors:
                        try:
                            # Get source sensor values from slot_values
                            source1_data = slot_values.get(virtual_sensor.source_sensor1)
                            source2_data = slot_values.get(virtual_sensor.source_sensor2)
                            
                            if source1_data is None or source2_data is None:
                                _Logger.debug(
                                    "Skipping virtual sensor '%s' for slot %s: source sensor(s) not available",
                                    virtual_sensor.name,
                                    slot_start,
                                )
                                continue
                            
                            value1, _ = source1_data
                            value2, _ = source2_data
                            
                            # Calculate virtual sensor value from resampled raw data
                            virtual_value = virtual_sensor.calculate(value1, value2)
                            
                            if virtual_value is not None:
                                resampled = ResampledSample(
                                    slot_start=slot_start,
                                    category=virtual_sensor.name,
                                    value=virtual_value,
                                    unit=virtual_sensor.unit,
                                    is_derived=True,  # Virtual/derived sensor data
                                )
                                session.add(resampled)
                                _Logger.debug(
                                    "Virtual sensor '%s' calculated: %.2f %s (from %.2f and %.2f)",
                                    virtual_sensor.name,
                                    virtual_value,
                                    virtual_sensor.unit,
                                    value1,
                                    value2,
                                )
                        except Exception as e:
                            # Log error but don't fail the entire resampling transaction
                            _Logger.error(
                                "Error calculating virtual sensor '%s' for slot %s: %s",
                                virtual_sensor.name,
                                slot_start,
                                e,
                            )
                    
                    slots_saved += 1
                else:
                    slots_skipped += 1

                slot_start = slot_end

            # Step 6: Commit all changes
            session.commit()

            _Logger.info(
                "Resample completed successfully. Processed: %d, Saved: %d, Skipped: %d, Rate: %dm",
                slots_processed,
                slots_saved,
                slots_skipped,
                sample_rate_minutes,
            )

            return ResampleStats(
                slots_processed=slots_processed,
                slots_saved=slots_saved,
                slots_skipped=slots_skipped,
                categories=list(category_to_entity.keys()),
                start_time=effective_start,
                end_time=global_end,
                sample_rate_minutes=sample_rate_minutes,
                table_flushed=table_flushed,
            )

    except SQLAlchemyError as e:
        _Logger.error("Error during resampling: %s", e)
        raise
