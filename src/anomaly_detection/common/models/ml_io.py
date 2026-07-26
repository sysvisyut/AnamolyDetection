"""
Machine Learning I/O contracts.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from anomaly_detection.common.models.enums import AnomalyCategory


class ModelScore(BaseModel):
    """Base class for model scores (Boundary F)."""
    entity_id: str = Field(..., min_length=1, description="Propagated from input")
    event_id: str = Field(..., min_length=1, description="Propagated from input")
    model_id: str = Field(..., description="Identifies the producing model")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="0.0 = normal, 1.0 = anomalous")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence in the score")
    cold_start_flag: bool = Field(..., description="Propagated from SessionMetadata")
    top_contributing_features: List[str] = Field(..., max_length=5, description="Feature names with highest attribution")

    model_config = {"from_attributes": True}


class ProfilingOutput(ModelScore):
    """Output of the Behavioral Profiling Model."""
    model_id: Literal["bpm"] = Field("bpm", description="BPM model identifier")
    ambiguity_reason: Optional[str] = Field(None, description="Populated by M10 for cold-start entities")


class DetectionOutput(ModelScore):
    """Output of the Sequence Detection Model."""
    model_id: Literal["sdm"] = Field("sdm", description="SDM model identifier")


class UnifiedAnomalySignal(BaseModel):
    """Fused score representation (Boundary G)."""
    entity_id: str = Field(..., min_length=1, description="Propagated from input")
    event_id: str = Field(..., min_length=1, description="Propagated from input")
    fused_score: float = Field(..., ge=0.0, le=1.0, description="Weighted combination of bpm and sdm scores")
    is_anomaly: bool = Field(..., description="True if fused_score >= fusion_threshold")
    bpm_score: float = Field(..., ge=0.0, le=1.0, description="Preserved for observability")
    sdm_score: float = Field(..., ge=0.0, le=1.0, description="Preserved for observability")
    cold_start_flag: bool = Field(..., description="Logical OR of component cold start flags")
    contributing_features: List[str] = Field(..., description="Union of top features from both models; deduplicated")

    model_config = {"from_attributes": True}


class ClassificationOutput(BaseModel):
    """Output of the Anomaly Classifier (Boundary H).
    Named ClassificationResult in DATA_SCHEMA.md, mapped to ClassificationOutput per M02 prompt.
    """
    entity_id: str = Field(..., min_length=1, description="Propagated from input")
    event_id: str = Field(..., min_length=1, description="Propagated from input")
    predicted_class: AnomalyCategory = Field(..., description="The most likely attack category")
    class_probabilities: Dict[str, float] = Field(..., description="Full posterior distribution over attack classes")
    classification_confidence: float = Field(..., ge=0.0, le=1.0, description="The winning class's probability")
    is_anomaly: bool = Field(..., description="Propagated from UnifiedAnomalySignal.is_anomaly")

    model_config = {"from_attributes": True}


class FeatureAttribution(BaseModel):
    """Individual feature contribution record."""
    feature_name: str = Field(..., description="Which feature contributed")
    feature_value: float = Field(..., description="Actual normalized value at inference time")
    attribution_score: float = Field(..., description="Signed; magnitude indicates importance")
    direction: str = Field(..., description='"toward_anomaly" or "toward_normal"')
    source_model: str = Field(..., description='"bpm", "sdm" or "bpm+sdm"')
    human_label: str = Field(..., description="Plain-English feature description")

    model_config = {"from_attributes": True}


class Explanation(BaseModel):
    """M09 output containing narrative and attributions."""
    narrative: str = Field(..., description="Human-readable explanation sentence")
    feature_attributions: List[FeatureAttribution] = Field(..., description="Ranked attributions")
    predicted_category: AnomalyCategory = Field(..., description="The predicted attack category")
    consistency_check_passed: bool = Field(..., description="Result of the consistency validator")
    is_ambiguous: bool = Field(..., description="True for ambiguous cases like insider drift")
    ambiguity_reason: Optional[str] = Field(None, description="Human-readable reason for ambiguity")

    model_config = {"from_attributes": True}
