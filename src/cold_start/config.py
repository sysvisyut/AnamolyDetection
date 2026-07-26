"""
Configuration for the Cold-Start Handler (M10).
"""

from pydantic import BaseModel, Field


class ColdStartConfig(BaseModel):
    """
    Configuration parameters for cold-start handling.
    """
    
    # Justification: 10 events provide a minimal statistical sample to compute a personalized 
    # mean and standard deviation. It also ensures that "Late Joiner" entities starting 
    # on Day 26 can graduate within the 5-day evaluation window given the expected event rate.
    graduation_threshold: int = Field(
        10, 
        ge=1, 
        description="Number of events required to graduate an entity from cold-start to warm."
    )
    
    # Justification: Discount reflects the inherent uncertainty of population-level priors.
    # The default value is defined in COLDSTART_DRIFT_STRATEGY.md to cap confidence at 0.5 
    # (assuming base confidence was 0.83, 0.83 * 0.6 = 0.5) or directly apply a flat discount 
    # to force it into the Ambiguity Zone. We default to 0.6 to effectively reduce 1.0 -> 0.6.
    confidence_discount_factor: float = Field(
        0.6, 
        gt=0.0, 
        le=1.0, 
        description="Factor multiplied with the BPM confidence for cold-start entities."
    )
    
    ambiguity_threshold: float = Field(
        0.65, 
        ge=0.0, 
        le=1.0, 
        description="Confidence below which an explanation is considered ambiguous."
    )
    
    event_counter_persistence_path: str = Field(
        "data/cold_start/event_counter.json",
        description="Path to the JSON file where the event counter state is persisted."
    )
