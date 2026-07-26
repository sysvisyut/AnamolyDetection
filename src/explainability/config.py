"""
Explainability Configuration (M09).
"""

from typing import Dict, Any
from pydantic import BaseModel, Field


class ExplainabilityConfig(BaseModel):
    """
    Configuration parameters for the Explainability Layer.
    """
    # Number of top features to include in the narrative
    top_n_features: int = Field(default=3, ge=1, le=4)
    
    # Threshold for filtering attribution features based on magnitude
    attribution_threshold: float = Field(default=0.05, ge=0.0)
    
    # Threshold for validating expected primary features vs top cited features
    consistency_threshold: float = Field(default=0.33, ge=0.0, le=1.0)
    
    # Threshold for classification confidence below which the explanation is considered ambiguous
    ambiguity_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    
    # Number of steps for Captum Integrated Gradients
    ig_n_steps: int = Field(default=50, ge=10)
    
    # Hardcoded limits
    max_narrative_length: int = Field(default=500, description="Max characters for narrative")
