"""
Resampling logic for aggregating raw sensor samples into 5-minute time slots.

This module provides functionality to:
1. Map logical categories to Home Assistant entities
2. Compute time-weighted averages for sensor data
3. Resample raw samples into uniform 5-minute time slots
"""

import logging
from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db import ResampledSample, Sample, SensorMapping
from db.core import engine, init_db_schema

_Logger = logging.getLogger(__name__)

# Fixed step size for resampling
RESAMPLE_STEP = timedelta(minutes=5)


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
    """
    aligned_minutes = (dt.minute // 5) * 5
    return dt.replace(minute=aligned_minutes, second=0, microsecond=0)


def resample_all_categories_to_5min() -> None:
    """
    Resample raw sensor samples into 5-minute slots for all configured categories.

    This function:
    1. Ensures DB schema exists (creates tables if missing).
    2. Fetches category-to-entity mappings.
    3. Computes the global time range where all categories have data.
    4. Iterates over 5-minute slots and computes time-weighted averages.
    5. Only writes complete slots (all categories have values).
    6. Ensures idempotence by deleting existing rows before inserting.
    """
    # Step 1: Ensure DB schema exists
    init_db_schema()

    # Step 2: Fetch mappings
    category_to_entity = get_primary_entities_by_category()

    if not category_to_entity:
        _Logger.warning(
            "No active sensor mappings configured, skipping resample."
        )
        return

    # Step 3: Get global time range
    global_start, global_end, category_to_entity = get_global_range_for_all_categories()

    if global_start is None or global_end is None:
        _Logger.warning(
            "No global time range available for all categories, skipping resample."
        )
        return

    # Step 4: Align global_start to 5-minute boundary
    aligned_start = _align_to_5min_boundary(global_start)

    _Logger.info(
        "Starting resample: aligned_start=%s, global_end=%s, categories=%s",
        aligned_start,
        global_end,
        list(category_to_entity.keys()),
    )

    # Step 5: Iterate over slots
    try:
        with Session(engine) as session:
            slot_start = aligned_start

            while slot_start < global_end:
                slot_end = slot_start + RESAMPLE_STEP

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

                    # Insert new rows for each category
                    for category, (avg, unit) in slot_values.items():
                        resampled = ResampledSample(
                            slot_start=slot_start,
                            category=category,
                            value=avg,
                            unit=unit,
                        )
                        session.add(resampled)

                slot_start = slot_end

            # Step 6: Commit all changes
            session.commit()

            _Logger.info("Resample completed successfully.")

    except SQLAlchemyError as e:
        _Logger.error("Error during resampling: %s", e)
