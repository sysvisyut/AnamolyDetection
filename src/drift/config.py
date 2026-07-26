"""Configuration for the gated EWMA profile updater."""

from pydantic import BaseModel, Field

DEFAULT_EWMA_ALPHA = 0.05
DEFAULT_GATING_THRESHOLD = 0.40


class DriftConfig(BaseModel):
    """Runtime parameters for safe concept-drift profile adaptation."""

    ewma_alpha: float = Field(
        DEFAULT_EWMA_ALPHA,
        gt=0.0,
        le=1.0,
        description="Weight assigned to the current feature vector in an EWMA update.",
    )
    gating_threshold: float = Field(
        DEFAULT_GATING_THRESHOLD,
        ge=0.0,
        le=1.0,
        description=(
            "Scores at or above this value are rejected so anomalous behavior "
            "cannot be learned into a profile."
        ),
    )
