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
            "max_workers": int or None,  # None or 0 = auto-calculate
            "max_combinations": int or None,  # None = default (1024)
        }
    """
    try:
        with Session(engine) as session:
            stmt = select(OptimizerConfig).order_by(OptimizerConfig.id.desc()).limit(1)
            config = session.scalars(stmt).first()
            
            if config:
                return {
                    "max_workers": config.max_workers,
                    "max_combinations": config.max_combinations,
                }
            else:
                # No config exists, return defaults
                return {
                    "max_workers": None,  # Auto-calculate
                    "max_combinations": None,  # Default (1024)
                }
    except Exception as e:
        _Logger.error("Error getting optimizer config: %s", e, exc_info=True)
        return {
            "max_workers": None,
            "max_combinations": None,
        }


def set_optimizer_config(
    max_workers: Optional[int] = None,
    max_combinations: Optional[int] = None,
) -> bool:
    """
    Set the optimizer configuration.
    
    Args:
        max_workers: Maximum number of workers (None or 0 = auto-calculate)
        max_combinations: Maximum feature combinations to test (None = default 1024)
        
    Returns:
        True if successfully saved, False otherwise
    """
    try:
        # Validate max_workers
        if max_workers is not None and max_workers < 0:
            _Logger.error("Invalid max_workers value: %d (must be >= 0 or None)", max_workers)
            return False
        
        # Validate max_combinations
        if max_combinations is not None and max_combinations < 1:
            _Logger.error("Invalid max_combinations value: %d (must be >= 1 or None)", max_combinations)
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
                # Always update max_workers (including None for auto-calculate)
                config.max_workers = max_workers
                if max_combinations is not None:
                    config.max_combinations = max_combinations
                config.updated_at = datetime.now(timezone.utc)
            else:
                # Create new config
                config = OptimizerConfig(
                    max_workers=max_workers,
                    max_combinations=max_combinations,
                    updated_at=datetime.now(timezone.utc),
                )
                session.add(config)
            
            session.commit()
            _Logger.info(
                "Optimizer config saved: max_workers=%s, max_combinations=%s",
                max_workers,
                max_combinations
            )
            return True
            
    except Exception as e:
        _Logger.error("Error setting optimizer config: %s", e, exc_info=True)
        return False
