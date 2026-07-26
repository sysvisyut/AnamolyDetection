"""Entities Router (M13)."""

from typing import List
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Query

from anomaly_detection.api.dependencies import get_profile_store, get_alert_store
from src.profiling.profile_store import ProfileStore
from anomaly_detection.stores.alert_store import AbstractAlertStore
from anomaly_detection.common.models.entities import EntityHistoryEntry

router = APIRouter()


class EntityStatusResponse(BaseModel):
    """Status summary for a single entity."""
    entity_id: str
    is_cold_start: bool
    profile_version: int
    drift_severity: str
    last_drift_check: str


@router.get("/{entity_id}/status", response_model=EntityStatusResponse)
async def get_entity_status(
    entity_id: str,
    profile_store: ProfileStore = Depends(get_profile_store)
):
    """Surfaces cold-start and drift status for the entity."""
    profile = profile_store.get_profile(entity_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Entity profile not found")
        
    drift_severity = "none"
    last_drift_check = "N/A"
    if profile.drift_metrics:
        drift_severity = profile.drift_metrics.drift_severity
        last_drift_check = profile.drift_metrics.last_drift_check

    return EntityStatusResponse(
        entity_id=profile.entity_id,
        is_cold_start=profile.cold_start_flag,
        profile_version=profile.profile_version,
        drift_severity=drift_severity,
        last_drift_check=last_drift_check
    )


@router.get("/{entity_id}/history", response_model=List[EntityHistoryEntry])
async def get_entity_history(
    entity_id: str,
    limit: int = Query(50, ge=1, le=1000),
    alert_store: AbstractAlertStore = Depends(get_alert_store)
):
    """Retrieves the chronological timeline of events for a specific entity."""
    # Assuming the underlying store has a get_entity_history method (as implemented in sqlite.py)
    # If the store interface doesn't have it, Python's duck typing will still execute the method 
    # if it exists, or raise AttributeError. We'll use hasattr to be safe.
    if not hasattr(alert_store, "get_entity_history"):
        raise HTTPException(status_code=501, detail="History retrieval not implemented for the current store")
        
    history = alert_store.get_entity_history(entity_id, limit=limit)
    return history
