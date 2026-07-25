"""
Entity profile and history models.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from anomaly_detection.common.models.enums import AnomalyCategory, EntityType


class DriftMetrics(BaseModel):
    """Drift metrics representation (T2 slot)."""
    feature_means_history: List[List[float]] = Field(..., description="Last K baseline_vector snapshots")
    last_drift_check: str = Field(..., description="ISO-8601 UTC")
    drift_severity: str = Field(..., description='Enum: "none", "low", "medium", "high"')
    drift_detected_at: Optional[str] = Field(None, description="ISO-8601 UTC of most recent detected drift")

    model_config = {"from_attributes": True}


class EntityProfile(BaseModel):
    """Entity profile representing the baseline behavior (Boundary E)."""
    entity_id: str = Field(..., description="Primary key")
    entity_type: EntityType = Field(..., description="user, service_account, or edge_device")
    baseline_vector: List[float] = Field(..., description="Rolling mean of feature_vector")
    baseline_std: List[float] = Field(..., description="Rolling standard deviation")
    sequence_history: List[List[float]] = Field(..., description="Last W feature_vector entries")
    most_frequent_country: str = Field(..., description="ISO 3166-1 alpha-2")
    known_mac_addresses: List[str] = Field(..., description="Set of MAC addresses seen")
    known_os_profiles: List[Dict[str, str]] = Field(..., description="Set of {os_family, os_version} dicts")
    known_protocols: List[str] = Field(..., description="Protocols seen for this entity")
    resource_access_counts: Dict[str, int] = Field(..., description="{resource_identifier: count}")
    command_frequency: Dict[str, int] = Field(..., description="{command: count}")
    event_count: int = Field(..., ge=0, description="Total events seen for this entity")
    cold_start_flag: bool = Field(..., description="True if event_count < MIN_PROFILE_EVENTS")
    
    # DriftMetrics is optional or empty dict in T1. We allow it to be optional or a dict.
    # To keep it strongly typed, we'll make it Optional[DriftMetrics] and default to None for T1.
    drift_metrics: Optional[DriftMetrics] = Field(None, description="Rolling distribution statistics")
    
    last_updated: str = Field(..., description="ISO-8601 UTC timestamp of last update")
    profile_version: int = Field(..., ge=1, description="Monotonically increasing version")

    model_config = {"from_attributes": True}


class EntityHistoryEntry(BaseModel):
    """A single event in an entity's history, typically used in dashboards."""
    event_id: str = Field(..., description="Event identifier")
    timestamp: str = Field(..., description="ISO-8601 UTC")
    resource_accessed: str = Field(..., description="Resource for this event")
    auth_outcome: str = Field(..., description='"success", "failure", "mfa_required"')
    
    risk_score: Optional[int] = Field(None, description="Alert risk score if an alert was generated")
    attack_class: AnomalyCategory = Field(..., description="Alert attack class or 'normal' if no alert")
    has_alert: bool = Field(..., description="True if this event generated an alert")

    model_config = {"from_attributes": True}
