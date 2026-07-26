"""Alerts Router (M13)."""

import asyncio
import json
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse

from anomaly_detection.api.dependencies import get_alert_store, get_alert_stream_queue
from anomaly_detection.common.models.alerts import AlertSummary, Alert
from anomaly_detection.stores.alert_store import AbstractAlertStore

router = APIRouter()


@router.get("", response_model=dict)
async def list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
    risk_tier: Optional[str] = None,
    attack_class: Optional[str] = None,
    entity_id: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    alert_store: AbstractAlertStore = Depends(get_alert_store)
):
    """Retrieve a ranked, paginated, and filterable queue of alerts."""
    risk_tiers = risk_tier.split(",") if risk_tier else None
    attack_classes = attack_class.split(",") if attack_class else None
    
    summaries, total_count = alert_store.get_alerts(
        page=page,
        page_size=page_size,
        risk_tier=risk_tiers,
        attack_class=attack_classes,
        entity_id=entity_id,
        since=since,
        until=until
    )
    
    return {
        "alerts": summaries,
        "total_count": total_count,
        "page": page,
        "page_size": page_size
    }


@router.get("/{alert_id}", response_model=Alert)
async def get_alert(
    alert_id: str,
    alert_store: AbstractAlertStore = Depends(get_alert_store)
):
    """Retrieve the full payload of a single alert."""
    alert = alert_store.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert
