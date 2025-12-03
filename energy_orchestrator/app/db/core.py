import os
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from db import Base

_Logger = logging.getLogger(__name__)

DB_HOST = os.environ.get("DB_HOST", "core-mariadb")
DB_USER = os.environ.get("DB_USER", "homeassistant")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "homeassistant")

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(DB_URL, future=True)


def test_db_connection() -> None:
    """Heel simpele check of MariaDB bereikbaar is."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _Logger.info("Verbinding met MariaDB geslaagd.")
    except SQLAlchemyError as e:
        _Logger.error("Fout bij verbinden met MariaDB: %s", e)


def init_db_schema() -> None:
    """Maak de tabellen aan als ze nog niet bestaan."""
    try:
        Base.metadata.create_all(engine)
        _Logger.info("Database schema bijgewerkt (samples).")
        
        # Run migrations for schema updates
        _migrate_add_is_derived_column()
        _migrate_add_optimizer_row_data_columns()
    except SQLAlchemyError as e:
        _Logger.error("Fout bij aanmaken schema in MariaDB: %s", e)


def _migrate_add_is_derived_column() -> None:
    """Add is_derived column to resampled_samples table if it doesn't exist.
    
    This migration adds a boolean column to distinguish between:
    - Raw sensor data (is_derived=False)
    - Derived/calculated data like virtual sensors (is_derived=True)
    
    Existing rows default to False (raw sensor data).
    """
    try:
        with engine.connect() as conn:
            # Check if column already exists
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'resampled_samples' 
                AND COLUMN_NAME = 'is_derived'
            """))
            exists = result.scalar() > 0
            
            if not exists:
                _Logger.info("Adding is_derived column to resampled_samples table...")
                conn.execute(text("""
                    ALTER TABLE resampled_samples 
                    ADD COLUMN is_derived TINYINT(1) NOT NULL DEFAULT 0
                """))
                conn.commit()
                _Logger.info("Successfully added is_derived column to resampled_samples table")
            else:
                _Logger.debug("is_derived column already exists in resampled_samples table")
    except SQLAlchemyError as e:
        _Logger.warning("Could not add is_derived column (may already exist): %s", e)


def _migrate_add_optimizer_row_data_columns() -> None:
    """Add first_row_json and last_row_json columns to optimizer_results table if they don't exist.
    
    This migration adds text columns to store the actual first and last rows
    of the training dataset for each optimizer result, providing visibility
    into the actual data used for training.
    """
    try:
        with engine.connect() as conn:
            # Check if first_row_json column exists
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'optimizer_results' 
                AND COLUMN_NAME = 'first_row_json'
            """))
            first_exists = result.scalar() > 0
            
            # Check if last_row_json column exists
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'optimizer_results' 
                AND COLUMN_NAME = 'last_row_json'
            """))
            last_exists = result.scalar() > 0
            
            if not first_exists:
                _Logger.info("Adding first_row_json column to optimizer_results table...")
                conn.execute(text("""
                    ALTER TABLE optimizer_results 
                    ADD COLUMN first_row_json TEXT NULL
                """))
                conn.commit()
                _Logger.info("Successfully added first_row_json column to optimizer_results table")
            else:
                _Logger.debug("first_row_json column already exists in optimizer_results table")
            
            if not last_exists:
                _Logger.info("Adding last_row_json column to optimizer_results table...")
                conn.execute(text("""
                    ALTER TABLE optimizer_results 
                    ADD COLUMN last_row_json TEXT NULL
                """))
                conn.commit()
                _Logger.info("Successfully added last_row_json column to optimizer_results table")
            else:
                _Logger.debug("last_row_json column already exists in optimizer_results table")
    except SQLAlchemyError as e:
        _Logger.warning("Could not add optimizer row data columns (may already exist): %s", e)
