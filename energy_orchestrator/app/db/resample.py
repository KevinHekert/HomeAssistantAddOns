"""Resample raw sensor samples into 5-minute slots per category.

This module provides functions to:
1. Determine primary entities per category from sensor_mappings
2. Compute global time range for all categories
3. Calculate time-weighted averages
4. Resample all categories to 5-minute intervals
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple

from sqlalchemy import func, and_
from sqlalchemy.orm import Session

from .models import Sample, SensorMapping, ResampledSample
from .schema import init_db_schema, get_session

logger = logging.getLogger(__name__)

# Fixed step size for resampling: 5 minutes
RESAMPLE_STEP = timedelta(minutes=5)


def get_primary_entities_by_category(session: Session | None = None) -> dict[str, str]:
    """Determine the primary entity_id per category from sensor_mappings.

    Only considers rows with is_active = TRUE. Sorts by:
    - category ASC
    - priority ASC
    - id ASC

    For each category, picks the first row as the primary entity.

    Args:
        session: Optional SQLAlchemy session. If not provided, creates a new one.

    Returns:
        Dictionary mapping category to its primary entity_id.
        Example: {"WIND": "sensor.knmi_windsnelheid", "OUTDOOR_TEMP": "sensor.knmi_temperatuur"}
    """
    close_session = False
    if session is None:
        session = get_session()
        close_session = True

    try:
        # Query active mappings, sorted by category, priority, and id
        mappings = (
            session.query(SensorMapping)
            .filter(SensorMapping.is_active == True)
            .order_by(
                SensorMapping.category.asc(),
                SensorMapping.priority.asc(),
                SensorMapping.id.asc(),
            )
            .all()
        )

        # Build the result dictionary, picking the first entity per category
        category_to_entity: dict[str, str] = {}
        for mapping in mappings:
            if mapping.category not in category_to_entity:
                category_to_entity[mapping.category] = mapping.entity_id

        if not category_to_entity:
            logger.warning("No active sensor mappings found in sensor_mappings table")

        return category_to_entity
    finally:
        if close_session:
            session.close()


def get_global_range_for_all_categories(
    session: Session | None = None,
) -> Tuple[datetime | None, datetime | None, dict[str, str]]:
    """Compute the global time range that covers all primary entities.

    For each primary entity:
    - Computes min_ts and max_ts from the samples table.

    Global range:
    - global_start = max(min_ts over all primary entities)
    - global_end = min(max_ts over all primary entities)

    If any primary entity has no data, returns (None, None, category_to_entity).

    Args:
        session: Optional SQLAlchemy session. If not provided, creates a new one.

    Returns:
        Tuple of (global_start, global_end, category_to_entity)
        where category_to_entity is the mapping of categories to their primary entities.
    """
    close_session = False
    if session is None:
        session = get_session()
        close_session = True

    try:
        category_to_entity = get_primary_entities_by_category(session)

        if not category_to_entity:
            logger.warning("No active sensor mappings, cannot compute global range")
            return (None, None, category_to_entity)

        global_start: datetime | None = None
        global_end: datetime | None = None

        for category, entity_id in category_to_entity.items():
            # Get min and max timestamps for this entity
            result = (
                session.query(
                    func.min(Sample.timestamp), func.max(Sample.timestamp)
                )
                .filter(Sample.entity_id == entity_id)
                .one()
            )

            min_ts, max_ts = result

            if min_ts is None or max_ts is None:
                logger.warning(
                    "Category '%s' (entity '%s') has no data in samples table",
                    category,
                    entity_id,
                )
                return (None, None, category_to_entity)

            # Update global range
            if global_start is None or min_ts > global_start:
                global_start = min_ts
            if global_end is None or max_ts < global_end:
                global_end = max_ts

        return (global_start, global_end, category_to_entity)
    finally:
        if close_session:
            session.close()


def compute_time_weighted_avg(
    session: Session,
    entity_id: str,
    window_start: datetime,
    window_end: datetime,
) -> Tuple[float | None, str | None]:
    """Compute time-weighted average for an entity within a time window.

    Uses piecewise constant (zero-order hold) interpolation:
    - Between two sample timestamps, value is the last known sample value.
    - Includes the last sample before window_start if available.

    Args:
        session: SQLAlchemy session.
        entity_id: The entity to compute average for.
        window_start: Start of the time window (inclusive).
        window_end: End of the time window (exclusive).

    Returns:
        Tuple of (average_value, unit) or (None, None) if no data available.
    """
    # Find the last sample before window_start
    prev_sample = (
        session.query(Sample)
        .filter(
            and_(
                Sample.entity_id == entity_id,
                Sample.timestamp < window_start,
            )
        )
        .order_by(Sample.timestamp.desc())
        .first()
    )

    # Get all samples within [window_start, window_end)
    samples_in_window = (
        session.query(Sample)
        .filter(
            and_(
                Sample.entity_id == entity_id,
                Sample.timestamp >= window_start,
                Sample.timestamp < window_end,
            )
        )
        .order_by(Sample.timestamp.asc())
        .all()
    )

    # If there's no previous sample and no samples in window, return None
    if prev_sample is None and not samples_in_window:
        return (None, None)

    # Initialize
    accum = 0.0
    total = 0.0
    current_time = window_start

    if prev_sample is not None:
        current_val = prev_sample.value
        current_unit = prev_sample.unit
    elif samples_in_window:
        # Use first sample in window
        current_val = samples_in_window[0].value
        current_unit = samples_in_window[0].unit
    else:
        # This case should not happen given the check above
        return (None, None)

    # Process samples in window
    for sample in samples_in_window:
        next_time = sample.timestamp
        dt = (next_time - current_time).total_seconds()

        if dt > 0:
            accum += current_val * dt
            total += dt

        current_time = next_time
        current_val = sample.value
        current_unit = sample.unit

    # Handle the remaining time until window_end
    dt_final = (window_end - current_time).total_seconds()
    if dt_final > 0:
        accum += current_val * dt_final
        total += dt_final

    if total == 0:
        return (None, None)

    avg = accum / total
    return (avg, current_unit)


def _align_to_5min_boundary(dt: datetime) -> datetime:
    """Align a datetime downwards to the nearest 5-minute boundary.

    Args:
        dt: The datetime to align.

    Returns:
        Datetime with minutes floored to multiple of 5, seconds/microseconds zeroed.
    """
    aligned_minutes = (dt.minute // 5) * 5
    return dt.replace(minute=aligned_minutes, second=0, microsecond=0)


def resample_all_categories_to_5min(db_path: str | None = None) -> None:
    """Resample all categories to 5-minute intervals.

    This function:
    1. Ensures DB schema exists
    2. Fetches primary entity mappings per category
    3. Computes the global time range
    4. Iterates over 5-minute slots
    5. For each slot, computes time-weighted average for all categories
    6. Only writes rows if all categories have valid values
    7. Uses DELETE + INSERT for idempotence

    Args:
        db_path: Optional path to the database file.
    """
    # Ensure DB schema exists
    init_db_schema(db_path)

    session = get_session()

    try:
        # Get primary entities by category
        category_to_entity = get_primary_entities_by_category(session)

        if not category_to_entity:
            logger.warning(
                "No active sensor mappings configured, skipping resample"
            )
            return

        # Get global time range
        global_start, global_end, _ = get_global_range_for_all_categories(session)

        if global_start is None or global_end is None:
            logger.warning(
                "No global range for all categories, skipping resample"
            )
            return

        # Align global_start to 5-minute boundary
        aligned_start = _align_to_5min_boundary(global_start)

        logger.info(
            "Resampling %d categories from %s to %s (aligned start: %s)",
            len(category_to_entity),
            global_start,
            global_end,
            aligned_start,
        )

        # Iterate over slots
        slot_start = aligned_start
        slots_processed = 0
        slots_written = 0

        while slot_start < global_end:
            slot_end = slot_start + RESAMPLE_STEP

            # Compute values for all categories
            slot_values: dict[str, Tuple[float, str | None]] = {}
            slot_complete = True

            for category, entity_id in category_to_entity.items():
                avg, unit = compute_time_weighted_avg(
                    session, entity_id, slot_start, slot_end
                )

                if avg is None:
                    # Slot is incomplete
                    slot_complete = False
                    break

                slot_values[category] = (avg, unit)

            if slot_complete:
                # Delete existing rows for this slot (idempotence)
                session.query(ResampledSample).filter(
                    ResampledSample.slot_start == slot_start
                ).delete()

                # Insert new rows
                for category, (avg, unit) in slot_values.items():
                    resampled = ResampledSample(
                        slot_start=slot_start,
                        category=category,
                        value=avg,
                        unit=unit,
                    )
                    session.add(resampled)

                slots_written += 1

            slots_processed += 1
            slot_start = slot_end

        # Commit all changes
        session.commit()

        logger.info(
            "Resampling complete: %d slots processed, %d slots written",
            slots_processed,
            slots_written,
        )

    except Exception as e:
        session.rollback()
        logger.error("Error during resampling: %s", e)
        raise
    finally:
        session.close()
