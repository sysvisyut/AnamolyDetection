"""
Feature-engineered representation models.
"""

from typing import List
from pydantic import BaseModel, Field, RootModel


class SessionMetadata(BaseModel):
    """Contextual flags for the session, not fed directly to models."""
    is_cold_start: bool = Field(..., description="True if entity has fewer than MIN_PROFILE_EVENTS historical events")
    delivery_mode_hint: str = Field(..., description="batch or simulated_stream; for FE internal use only")
    profile_event_count: int = Field(..., ge=0, description="How many historical events exist for this entity at inference time")

    model_config = {"from_attributes": True}


class EntityFeatureVector(RootModel[List[float]]):
    """
    Fixed-length feature vector output of the feature engineering pipeline.
    Length is exactly 24 dimensions per DATA_SCHEMA.md.
    """
    root: List[float] = Field(..., min_length=24, max_length=24)


class EntitySequence(RootModel[List[List[float]]]):
    """
    Variable-length sequence representation per DATA_SCHEMA.md's feature-engineered representation schema.
    Normally 20 vectors, but can be shorter (padded downstream) or empty.
    """
    root: List[List[float]]


class EngineeredFeatures(BaseModel):
    """Output of the feature engineering pipeline (Boundary C)."""
    entity_id: str = Field(..., description="propagated from InboundEvent; identifies the entity")
    event_id: str = Field(..., description="propagated from InboundEvent; the triggering event's ID")
    session_id: str = Field(..., description="propagated from InboundEvent; sequence grouping key")
    
    # We use the typed structures defined above
    feature_vector: EntityFeatureVector = Field(..., description="fixed-length normalized vector")
    sequence_window: EntitySequence = Field(..., description="sliding window of recent feature vectors")
    session_metadata: SessionMetadata = Field(..., description="non-numeric contextual flags")

    model_config = {"from_attributes": True}
