"""Database schema initialization for Energy Orchestrator."""

import os
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = "/data/energy_orchestrator.db"

_engine = None
_Session = None


def get_engine(db_path: str | None = None):
    """Get or create the SQLAlchemy engine.

    Args:
        db_path: Optional path to the SQLite database file.
                 If not provided, uses DEFAULT_DB_PATH or DB_PATH env var.

    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine

    if _engine is None:
        if db_path is None:
            db_path = os.environ.get("DB_PATH", DEFAULT_DB_PATH)
        db_url = f"sqlite:///{db_path}"
        _engine = create_engine(db_url, echo=False)
        logger.info("Created database engine for %s", db_path)

    return _engine


def get_session():
    """Get a new database session.

    Returns:
        SQLAlchemy Session instance.
    """
    global _Session

    if _Session is None:
        engine = get_engine()
        _Session = sessionmaker(bind=engine)

    return _Session()


def init_db_schema(db_path: str | None = None) -> None:
    """Initialize the database schema, creating all tables if they don't exist.

    This creates the following tables if missing:
    - samples
    - sensor_mappings
    - resampled_samples

    Args:
        db_path: Optional path to the SQLite database file.
    """
    global _engine, _Session

    # Reset engine and session if db_path is provided
    if db_path is not None:
        _engine = None
        _Session = None

    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    logger.info("Database schema initialized (all tables created if missing)")
