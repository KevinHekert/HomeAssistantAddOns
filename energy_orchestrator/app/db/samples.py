import logging
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db import Sample
from db.core import engine

_Logger = logging.getLogger(__name__)


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
    try:
        with Session(engine) as session:
            exists = (
                session.query(Sample.id)
                .filter(
                    Sample.entity_id == entity_id,
                    Sample.timestamp == timestamp,
                )
                .first()
                is not None
            )
        return exists
    except SQLAlchemyError as e:
        _Logger.error(
            "Fout bij controleren of sample bestaat voor %s @ %s: %s",
            entity_id,
            timestamp,
            e,
        )
        return True  # bij twijfel: liever niet dubbel inserten


def log_sample(entity_id: str, timestamp: datetime, value: float | None, unit: str | None) -> None:
    """Sla één generiek sample op in de database."""
    if value is None:
        _Logger.info("Geen waarde om op te slaan voor %s, overslaan.", entity_id)
        return

    try:
        with Session(engine) as session:
            sample = Sample(
                entity_id=entity_id,
                timestamp=timestamp,
                value=float(value),
                unit=unit,
            )
            session.add(sample)
            session.commit()
        _Logger.info(
            "Sample opgeslagen: entity=%s, ts=%s, value=%s, unit=%s",
            entity_id,
            timestamp,
            value,
            unit,
        )
    except SQLAlchemyError as e:
        _Logger.error("Fout bij opslaan van Sample voor %s: %s", entity_id, e)
