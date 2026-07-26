"""Configuration for the end-to-end inference orchestrator."""

from pydantic import BaseModel, Field

DEFAULT_ALERT_THRESHOLD = 0.5
DEFAULT_MAX_PROCESSING_LATENCY_MS = 75.0


class OrchestratorConfig(BaseModel):
    """Validated policy for alerting and inference latency observability."""

    alert_threshold: float = Field(DEFAULT_ALERT_THRESHOLD, ge=0.0, le=1.0)
    max_processing_latency_ms: float = Field(
        DEFAULT_MAX_PROCESSING_LATENCY_MS,
        gt=0.0,
        description="Target synchronous processing budget from STREAMING_ARCHITECTURE.md.",
    )
