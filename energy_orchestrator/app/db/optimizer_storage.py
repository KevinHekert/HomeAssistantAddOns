"""
Database operations for storing and retrieving optimizer results.

This module provides functions to:
- Save optimization runs and results to the database
- Retrieve saved optimization results
- Apply saved optimization configurations
"""

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from db import OptimizerRun, OptimizerResult
from db.core import engine
from ml.optimizer import OptimizerProgress, OptimizationResult

_Logger = logging.getLogger(__name__)


def save_optimizer_run(progress: OptimizerProgress) -> Optional[int]:
    """
    Save an optimizer run and all its results to the database.
    
    Args:
        progress: The OptimizerProgress object containing run data and results
        
    Returns:
        The ID of the saved run, or None if save failed
    """
    try:
        with Session(engine) as session:
            # Create the run record
            run = OptimizerRun(
                start_time=progress.start_time,
                end_time=progress.end_time,
                phase=progress.phase,
                total_configurations=progress.total_configurations,
                completed_configurations=progress.completed_configurations,
                error_message=progress.error_message,
            )
            session.add(run)
            session.flush()  # Get the run ID
            
            run_id = run.id
            best_result_db_id = None
            
            # Save all results
            for result in progress.results:
                result_record = OptimizerResult(
                    run_id=run_id,
                    config_name=result.config_name,
                    model_type=result.model_type,
                    experimental_features_json=json.dumps(result.experimental_features),
                    val_mape_pct=result.val_mape_pct,
                    val_mae_kwh=result.val_mae_kwh,
                    val_r2=result.val_r2,
                    train_samples=result.train_samples,
                    val_samples=result.val_samples,
                    success=result.success,
                    error_message=result.error_message,
                    training_timestamp=result.training_timestamp,
                )
                session.add(result_record)
                session.flush()
                
                # Track the best result
                if progress.best_result and result is progress.best_result:
                    best_result_db_id = result_record.id
            
            # Update the run with the best result ID
            if best_result_db_id:
                run.best_result_id = best_result_db_id
            
            session.commit()
            _Logger.info("Saved optimizer run %d with %d results", run_id, len(progress.results))
            return run_id
            
    except Exception as e:
        _Logger.error("Error saving optimizer run: %s", e, exc_info=True)
        return None


def get_latest_optimizer_run() -> Optional[dict]:
    """
    Get the most recent optimizer run with its results.
    
    Returns:
        Dictionary with run metadata and results, or None if no runs found
    """
    try:
        with Session(engine) as session:
            # Get the latest run
            stmt = select(OptimizerRun).order_by(OptimizerRun.start_time.desc()).limit(1)
            run = session.scalars(stmt).first()
            
            if not run:
                return None
            
            # Get all results for this run
            results_stmt = select(OptimizerResult).where(
                OptimizerResult.run_id == run.id
            ).order_by(OptimizerResult.id)
            results = session.scalars(results_stmt).all()
            
            # Find best result
            best_result = None
            if run.best_result_id:
                for result in results:
                    if result.id == run.best_result_id:
                        best_result = _result_to_dict(result)
                        break
            
            return {
                "id": run.id,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "phase": run.phase,
                "total_configurations": run.total_configurations,
                "completed_configurations": run.completed_configurations,
                "error_message": run.error_message,
                "best_result": best_result,
                "results": [_result_to_dict(r) for r in results],
            }
            
    except Exception as e:
        _Logger.error("Error retrieving latest optimizer run: %s", e, exc_info=True)
        return None


def get_optimizer_run_by_id(run_id: int) -> Optional[dict]:
    """
    Get a specific optimizer run by ID with its results.
    
    Args:
        run_id: The ID of the run to retrieve
        
    Returns:
        Dictionary with run metadata and results, or None if not found
    """
    try:
        with Session(engine) as session:
            run = session.get(OptimizerRun, run_id)
            
            if not run:
                return None
            
            # Get all results for this run
            results_stmt = select(OptimizerResult).where(
                OptimizerResult.run_id == run.id
            ).order_by(OptimizerResult.id)
            results = session.scalars(results_stmt).all()
            
            # Find best result
            best_result = None
            if run.best_result_id:
                for result in results:
                    if result.id == run.best_result_id:
                        best_result = _result_to_dict(result)
                        break
            
            return {
                "id": run.id,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "phase": run.phase,
                "total_configurations": run.total_configurations,
                "completed_configurations": run.completed_configurations,
                "error_message": run.error_message,
                "best_result": best_result,
                "results": [_result_to_dict(r) for r in results],
            }
            
    except Exception as e:
        _Logger.error("Error retrieving optimizer run %d: %s", run_id, e, exc_info=True)
        return None


def get_optimizer_result_by_id(result_id: int) -> Optional[dict]:
    """
    Get a specific optimizer result by ID.
    
    Args:
        result_id: The ID of the result to retrieve
        
    Returns:
        Dictionary with result data, or None if not found
    """
    try:
        with Session(engine) as session:
            result = session.get(OptimizerResult, result_id)
            
            if not result:
                return None
            
            return _result_to_dict(result)
            
    except Exception as e:
        _Logger.error("Error retrieving optimizer result %d: %s", result_id, e, exc_info=True)
        return None


def list_optimizer_runs(limit: int = 10) -> list[dict]:
    """
    List recent optimizer runs (summary only, without full results).
    
    Args:
        limit: Maximum number of runs to return
        
    Returns:
        List of run summaries
    """
    try:
        with Session(engine) as session:
            stmt = select(OptimizerRun).order_by(
                OptimizerRun.start_time.desc()
            ).limit(limit)
            runs = session.scalars(stmt).all()
            
            summaries = []
            for run in runs:
                # Get best result summary if available
                best_result_summary = None
                if run.best_result_id:
                    best_result = session.get(OptimizerResult, run.best_result_id)
                    if best_result:
                        best_result_summary = {
                            "id": best_result.id,
                            "config_name": best_result.config_name,
                            "model_type": best_result.model_type,
                            "val_mape_pct": best_result.val_mape_pct,
                        }
                
                summaries.append({
                    "id": run.id,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "end_time": run.end_time.isoformat() if run.end_time else None,
                    "phase": run.phase,
                    "total_configurations": run.total_configurations,
                    "completed_configurations": run.completed_configurations,
                    "best_result": best_result_summary,
                })
            
            return summaries
            
    except Exception as e:
        _Logger.error("Error listing optimizer runs: %s", e, exc_info=True)
        return []


def _result_to_dict(result: OptimizerResult) -> dict:
    """Convert an OptimizerResult model to a dictionary."""
    return {
        "id": result.id,
        "run_id": result.run_id,
        "config_name": result.config_name,
        "model_type": result.model_type,
        "experimental_features": json.loads(result.experimental_features_json),
        "val_mape_pct": result.val_mape_pct,
        "val_mae_kwh": result.val_mae_kwh,
        "val_r2": result.val_r2,
        "train_samples": result.train_samples,
        "val_samples": result.val_samples,
        "success": result.success,
        "error_message": result.error_message,
        "training_timestamp": result.training_timestamp.isoformat() if result.training_timestamp else None,
    }
