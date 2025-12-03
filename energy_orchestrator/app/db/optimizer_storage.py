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


def create_optimizer_run(
    start_time: datetime,
    total_configurations: int,
) -> Optional[int]:
    """
    Create a new optimizer run record in the database.
    
    This is used to initialize a run before streaming results.
    
    Args:
        start_time: When the optimization started
        total_configurations: Total number of configurations to test
        
    Returns:
        The ID of the created run, or None if creation failed
    """
    try:
        with Session(engine) as session:
            run = OptimizerRun(
                start_time=start_time,
                end_time=None,
                phase="initializing",
                total_configurations=total_configurations,
                completed_configurations=0,
                error_message=None,
            )
            session.add(run)
            session.commit()
            _Logger.info("Created optimizer run %d with %d total configurations", run.id, total_configurations)
            return run.id
            
    except Exception as e:
        _Logger.error("Error creating optimizer run: %s", e, exc_info=True)
        return None


def save_optimizer_result(
    run_id: int,
    result: OptimizationResult,
) -> Optional[int]:
    """
    Save a single optimizer result to the database (streaming mode).
    
    This allows results to be saved immediately as they complete,
    rather than keeping all results in memory.
    
    Args:
        run_id: The ID of the run this result belongs to
        result: The OptimizationResult to save
        
    Returns:
        The ID of the saved result, or None if save failed
    """
    try:
        with Session(engine) as session:
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
            session.commit()
            return result_record.id
            
    except Exception as e:
        _Logger.error("Error saving optimizer result: %s", e, exc_info=True)
        return None


def update_optimizer_run_progress(
    run_id: int,
    completed_configurations: int,
    phase: str,
    best_result_id: Optional[int] = None,
) -> bool:
    """
    Update the progress of an optimizer run.
    
    Args:
        run_id: The ID of the run to update
        completed_configurations: Number of completed configurations
        phase: Current phase (e.g., "training", "complete")
        best_result_id: Optional ID of the best result so far
        
    Returns:
        True if update succeeded, False otherwise
    """
    try:
        with Session(engine) as session:
            stmt = (
                update(OptimizerRun)
                .where(OptimizerRun.id == run_id)
                .values(
                    completed_configurations=completed_configurations,
                    phase=phase,
                    best_result_id=best_result_id if best_result_id else OptimizerRun.best_result_id,
                )
            )
            session.execute(stmt)
            session.commit()
            return True
            
    except Exception as e:
        _Logger.error("Error updating optimizer run progress: %s", e, exc_info=True)
        return False


def complete_optimizer_run(
    run_id: int,
    end_time: datetime,
    phase: str = "complete",
    error_message: Optional[str] = None,
) -> bool:
    """
    Mark an optimizer run as complete.
    
    Args:
        run_id: The ID of the run to complete
        end_time: When the optimization ended
        phase: Final phase (e.g., "complete" or "error")
        error_message: Optional error message if phase is "error"
        
    Returns:
        True if update succeeded, False otherwise
    """
    try:
        with Session(engine) as session:
            stmt = (
                update(OptimizerRun)
                .where(OptimizerRun.id == run_id)
                .values(
                    end_time=end_time,
                    phase=phase,
                    error_message=error_message,
                )
            )
            session.execute(stmt)
            session.commit()
            _Logger.info("Completed optimizer run %d with phase '%s'", run_id, phase)
            return True
            
    except Exception as e:
        _Logger.error("Error completing optimizer run: %s", e, exc_info=True)
        return False


def get_optimizer_run_best_result(run_id: int) -> Optional[dict]:
    """
    Get the best result for a specific optimizer run.
    
    Args:
        run_id: The ID of the run
        
    Returns:
        Dictionary with best result data, or None if not found
    """
    try:
        with Session(engine) as session:
            # Get the run to find best_result_id
            run = session.get(OptimizerRun, run_id)
            if not run or not run.best_result_id:
                return None
            
            # Get the best result
            result = session.get(OptimizerResult, run.best_result_id)
            if not result:
                return None
            
            return _result_to_dict(result)
            
    except Exception as e:
        _Logger.error("Error getting best result for run %d: %s", run_id, e, exc_info=True)
        return None


def get_optimizer_run_top_results(run_id: int, limit: int = 20) -> list[dict]:
    """
    Get the top N results for a specific optimizer run, sorted by MAPE.
    
    Args:
        run_id: The ID of the run
        limit: Maximum number of results to return
        
    Returns:
        List of result dictionaries sorted by val_mape_pct (ascending)
    """
    try:
        with Session(engine) as session:
            stmt = (
                select(OptimizerResult)
                .where(OptimizerResult.run_id == run_id)
                .where(OptimizerResult.success == True)
                .where(OptimizerResult.val_mape_pct.isnot(None))
                .order_by(OptimizerResult.val_mape_pct.asc())
                .limit(limit)
            )
            results = session.scalars(stmt).all()
            
            return [_result_to_dict(r) for r in results]
            
    except Exception as e:
        _Logger.error("Error getting top results for run %d: %s", run_id, e, exc_info=True)
        return []


def save_optimizer_run(progress: OptimizerProgress) -> Optional[int]:
    """
    Save an optimizer run (legacy function - kept for backward compatibility).
    
    NOTE: This function is now primarily for saving runs that weren't
    already saved via streaming. New code should use create_optimizer_run()
    and save_optimizer_result() for streaming saves.
    
    Args:
        progress: The OptimizerProgress object containing run data
        
    Returns:
        The ID of the saved run (or existing run_id if already saved), or None if save failed
    """
    # If run was already saved via streaming, just return the run_id
    if progress.run_id is not None:
        _Logger.info("Optimizer run %d already saved via streaming", progress.run_id)
        return progress.run_id
    
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
            
            # Update the run with the best result ID if available
            if progress.best_result_db_id:
                run.best_result_id = progress.best_result_db_id
            
            session.commit()
            _Logger.info("Saved optimizer run %d (no results in memory - use streaming instead)", run_id)
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
