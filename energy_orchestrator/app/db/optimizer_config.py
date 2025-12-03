"""
Database operations for storing and retrieving optimizer configuration.

This module provides functions to:
- Get and set optimizer configuration settings (max_workers)
- Initialize default configuration
"""

import logging
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from db import OptimizerConfig
from db.core import engine

_Logger = logging.getLogger(__name__)


def get_optimizer_config() -> dict:
    """
    Get the current optimizer configuration.
    
    Returns:
        Dictionary with optimizer configuration:
        {
            "max_workers": int or None  # None or 0 = auto-calculate
        }
    """
    try:
        with Session(engine) as session:
            stmt = select(OptimizerConfig).order_by(OptimizerConfig.id.desc()).limit(1)
            config = session.scalars(stmt).first()
            
            if config:
                return {
                    "max_workers": config.max_workers,
                }
            else:
                # No config exists, return defaults
                return {
                    "max_workers": None,  # Auto-calculate
                }
    except Exception as e:
        _Logger.error("Error getting optimizer config: %s", e, exc_info=True)
        return {
            "max_workers": None,
        }


def set_optimizer_config(max_workers: Optional[int] = None) -> bool:
    """
    Set the optimizer configuration.
    
    Args:
        max_workers: Maximum number of workers (None or 0 = auto-calculate)
        
    Returns:
        True if successfully saved, False otherwise
    """
    try:
        # Validate max_workers
        if max_workers is not None and max_workers < 0:
            _Logger.error("Invalid max_workers value: %d (must be >= 0 or None)", max_workers)
            return False
        
        # Convert 0 to None for consistency
        if max_workers == 0:
            max_workers = None
        
        with Session(engine) as session:
            # Get or create config record
            stmt = select(OptimizerConfig).order_by(OptimizerConfig.id.desc()).limit(1)
            config = session.scalars(stmt).first()
            
            if config:
                # Update existing config
                config.max_workers = max_workers
                config.updated_at = datetime.now(timezone.utc)
            else:
                # Create new config
                config = OptimizerConfig(
                    max_workers=max_workers,
                    updated_at=datetime.now(timezone.utc),
                )
                session.add(config)
            
            session.commit()
            _Logger.info("Optimizer config saved: max_workers=%s", max_workers)
            return True
            
    except Exception as e:
        _Logger.error("Error setting optimizer config: %s", e, exc_info=True)
        return False
