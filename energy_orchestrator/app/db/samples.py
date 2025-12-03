import logging
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db import Sample, ResampledSample
from db.core import engine

_Logger = logging.getLogger(__name__)


def _normalize_timestamp(ts: datetime) -> datetime:
    """
    Normalize a timestamp by stripping microseconds.

    Preserves the actual timestamp from Home Assistant, only removing
    microsecond precision which is not needed for sample storage.
    """
    return ts.replace(microsecond=0)


def get_latest_sample_timestamp(entity_id: str) -> datetime | None:
    """Geef de laatste timestamp terug voor deze entiteit, of None als er geen samples zijn."""
    try:
        with Session(engine) as session:
            result = (
                session.query(Sample.timestamp)
                .filter(Sample.entity_id == entity_id)
                .order_by(Sample.timestamp.desc())
                .limit(1)
                .one_or_none()
            )
        if result is None:
            _Logger.info("Nog geen samples gevonden voor %s.", entity_id)
            return None

        latest_ts = result[0]
        _Logger.info("Laatste sample voor %s: %s", entity_id, latest_ts)
        return latest_ts
    except SQLAlchemyError as e:
        _Logger.error("Fout bij ophalen laatste sample voor %s: %s", entity_id, e)
        return None


def sample_exists(entity_id: str, timestamp: datetime) -> bool:
    """Check of er al een sample is voor deze entity + timestamp."""
    normalized_ts = _normalize_timestamp(timestamp)
    try:
        with Session(engine) as session:
            exists = (
                session.query(Sample.id)
                .filter(
                    Sample.entity_id == entity_id,
                    Sample.timestamp == normalized_ts,
                )
                .first()
                is not None
            )
        return exists
    except SQLAlchemyError as e:
        _Logger.error(
            "Fout bij controleren of sample bestaat voor %s @ %s: %s",
            entity_id,
            normalized_ts,
            e,
        )
        return True  # bij twijfel: liever niet dubbel inserten


def get_sensor_info() -> list[dict]:
    """
    Get first and last timestamp for each unique entity_id in the samples table.
    
    Returns:
        List of dicts with entity_id, first_timestamp, last_timestamp, and sample_count
    """
    try:
        with Session(engine) as session:
            result = (
                session.query(
                    Sample.entity_id,
                    func.min(Sample.timestamp).label("first_timestamp"),
                    func.max(Sample.timestamp).label("last_timestamp"),
                    func.count(Sample.id).label("sample_count"),
                )
                .group_by(Sample.entity_id)
                .order_by(Sample.entity_id)
                .all()
            )
            return [
                {
                    "entity_id": row.entity_id,
                    "first_timestamp": row.first_timestamp.isoformat() if row.first_timestamp else None,
                    "last_timestamp": row.last_timestamp.isoformat() if row.last_timestamp else None,
                    "sample_count": row.sample_count,
                }
                for row in result
            ]
    except SQLAlchemyError as e:
        _Logger.error("Error getting sensor info: %s", e)
        return []


def get_resampled_sensor_info() -> list[dict]:
    """
    Get first and last slot_start timestamp for each unique category in the resampled_samples table.
    
    Returns:
        List of dicts with category, unit, is_derived, first_timestamp, last_timestamp, and sample_count
    """
    try:
        with Session(engine) as session:
            result = (
                session.query(
                    ResampledSample.category,
                    ResampledSample.unit,
                    ResampledSample.is_derived,
                    func.min(ResampledSample.slot_start).label("first_timestamp"),
                    func.max(ResampledSample.slot_start).label("last_timestamp"),
                    func.count(ResampledSample.id).label("sample_count"),
                )
                .group_by(ResampledSample.category, ResampledSample.unit, ResampledSample.is_derived)
                .order_by(ResampledSample.category)
                .all()
            )
            return [
                {
                    "category": row.category,
                    "unit": row.unit,
                    "is_derived": row.is_derived,
                    "first_timestamp": row.first_timestamp.isoformat() if row.first_timestamp else None,
                    "last_timestamp": row.last_timestamp.isoformat() if row.last_timestamp else None,
                    "sample_count": row.sample_count,
                }
                for row in result
            ]
    except SQLAlchemyError as e:
        _Logger.error("Error getting resampled sensor info: %s", e)
        return []


def log_sample(entity_id: str, timestamp: datetime, value: float | None, unit: str | None) -> None:
    """
    Store or update a sample in the database.

    Normalizes timestamp by stripping microseconds and uses upsert logic:
    - If a sample already exists for this entity_id + timestamp, update it
    - Otherwise, create a new sample
    """
    if value is None:
        _Logger.info("Geen waarde om op te slaan voor %s, overslaan.", entity_id)
        return

    normalized_ts = _normalize_timestamp(timestamp)

    try:
        with Session(engine) as session:
            # Try to find existing sample
            existing = (
                session.query(Sample)
                .filter(
                    Sample.entity_id == entity_id,
                    Sample.timestamp == normalized_ts,
                )
                .first()
            )

            if existing:
                # Update existing sample
                existing.value = float(value)
                existing.unit = unit
                _Logger.debug(
                    "Sample bijgewerkt: entity=%s, ts=%s, value=%s, unit=%s",
                    entity_id,
                    normalized_ts,
                    value,
                    unit,
                )
            else:
                # Create new sample
                sample = Sample(
                    entity_id=entity_id,
                    timestamp=normalized_ts,
                    value=float(value),
                    unit=unit,
                )
                session.add(sample)
                _Logger.debug(
                    "Sample opgeslagen: entity=%s, ts=%s, value=%s, unit=%s",
                    entity_id,
                    normalized_ts,
                    value,
                    unit,
                )
            session.commit()
    except SQLAlchemyError as e:
        _Logger.error("Fout bij opslaan van Sample voor %s: %s", entity_id, e)
