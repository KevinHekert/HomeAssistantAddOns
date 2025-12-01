import logging
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db import SyncStatus
from db.core import engine

_Logger = logging.getLogger(__name__)


def update_sync_attempt(entity_id: str, attempt_ts: datetime, success: bool) -> None:
    """Registreer een sync-poging (en optioneel succesvolle sync) voor een entity."""
    try:
        with Session(engine) as session:
            state = session.get(SyncStatus, entity_id)
            if state is None:
                state = SyncStatus(entity_id=entity_id)

            state.last_attempt = attempt_ts
            if success:
                state.last_success = attempt_ts

            session.add(state)
            session.commit()
        _Logger.info(
            "SyncStatus bijgewerkt voor %s: last_attempt=%s, success=%s",
            entity_id,
            attempt_ts,
            success,
        )
    except SQLAlchemyError as e:
        _Logger.error(
            "Fout bij bijwerken van SyncStatus voor %s: %s", entity_id, e
        )


def get_sync_status(entity_id: str) -> SyncStatus | None:
    """Geef SyncStatus terug voor een entity (of None als die nog niet bestaat)."""
    try:
        with Session(engine) as session:
            return session.get(SyncStatus, entity_id)
    except SQLAlchemyError as e:
        _Logger.error(
            "Fout bij ophalen van SyncStatus voor %s: %s", entity_id, e
        )
        return None
