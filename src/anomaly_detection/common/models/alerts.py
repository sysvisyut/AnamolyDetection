"""
Alert and explainability models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from anomaly_detection.common.models.enums import AnomalyCategory


class FeatureAttribution(BaseModel):
    """Individual feature contribution record."""
    feature_name: str = Field(..., description="Which feature contributed")
    feature_value: float = Field(..., description="Actual normalized value at inference time")
    attribution_score: float = Field(..., description="Signed; magnitude indicates importance")
    direction: str = Field(..., description='"toward_anomaly" or "toward_normal"')
    source_model: str = Field(..., description='"bpm" or "sdm"')
    human_label: str = Field(..., description="Plain-English feature description")

    model_config = {"from_attributes": True}


class RiskScore(BaseModel):
    """Composite risk score representation."""
    risk_score: int = Field(..., ge=0, le=100, description="Composite risk score [0, 100]")
    risk_tier: str = Field(..., description='Enum: "low", "medium", "high", "critical"')

    model_config = {"from_attributes": True}


class Explanation(BaseModel):
    """Explanation and feature attribution grouping."""
    human_readable_explanation: str = Field(..., max_length=500, description="Natural language explanation")
    feature_attributions: List[FeatureAttribution] = Field(..., min_length=1, max_length=10, description="Ordered by descending magnitude")

    model_config = {"from_attributes": True}


class Alert(BaseModel):
    """
    The full alert object (corresponds to AlertPayload in DATA_SCHEMA.md).
    Updated per M02 prompt to nest RiskScore and Explanation.
    """
    alert_id: str = Field(..., description="UUID v4; primary key")
    entity_id: str = Field(..., min_length=1, description="Entity that triggered the alert")
    event_id: str = Field(..., min_length=1, description="The specific event that triggered scoring")
    session_id: str = Field(..., min_length=1, description="Session containing the triggering event")
    timestamp: str = Field(..., description="ISO-8601 UTC of the event")
    detected_at: str = Field(..., description="ISO-8601 UTC when the alert was generated")
    
    risk: RiskScore = Field(..., description="Nested risk score object")
    explanation: Explanation = Field(..., description="Nested explanation object")

    attack_class: AnomalyCategory = Field(..., description="Predicted attack category")
    classification_confidence: float = Field(..., ge=0.0, le=1.0, description="Classifier confidence")
    fused_score: float = Field(..., ge=0.0, le=1.0, description="Raw fused anomaly score")
    bpm_score: float = Field(..., ge=0.0, le=1.0, description="BPM component score")
    sdm_score: float = Field(..., ge=0.0, le=1.0, description="SDM component score")
    cold_start_flag: bool = Field(..., description="True if entity had insufficient history")
    
    raw_event_snapshot: Dict[str, Any] = Field(..., description="Serialized InboundEvent minus delivery_mode")
    
    analyst_decision: Optional[str] = Field(None, description='Enum: "true_positive", "false_positive", "needs_review"')
    analyst_notes: Optional[str] = Field(None, max_length=2000, description="Analyst annotation (T3)")

    model_config = {"from_attributes": True}
