"""Continuous, weighted fusion of BPM and SDM anomaly scores."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from anomaly_detection.common.models.ml_io import (
    DetectionOutput,
    ProfilingOutput,
    UnifiedAnomalySignal,
)

DEFAULT_BPM_WEIGHT = 0.5
DEFAULT_FUSION_THRESHOLD = 0.5
DEFAULT_SDM_WEIGHT = 0.5


class FusionConfig(BaseModel):
    """Validated policy for combining model scores into boundary G."""

    bpm_weight: float = Field(DEFAULT_BPM_WEIGHT, ge=0.0, le=1.0)
    sdm_weight: float = Field(DEFAULT_SDM_WEIGHT, ge=0.0, le=1.0)
    fusion_threshold: float = Field(DEFAULT_FUSION_THRESHOLD, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights(self) -> FusionConfig:
        """Require the two score weights to form a convex combination."""
        if abs((self.bpm_weight + self.sdm_weight) - 1.0) > 1e-9:
            raise ValueError("bpm_weight and sdm_weight must sum to 1.0")
        return self


class ScoreFusion:
    """Create a unified anomaly signal without collapsing score continuity."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        """Initialize fusion with a validated, injectable configuration."""
        self.config = config or FusionConfig()

    def fuse(
        self,
        profiling_output: ProfilingOutput,
        detection_output: DetectionOutput,
    ) -> UnifiedAnomalySignal:
        """Combine matching BPM and SDM scores into a boundary-G signal.

        Raises:
            ValueError: If the two model outputs do not refer to the same event.
        """
        self._validate_matching_ids(profiling_output, detection_output)
        fused_score = (
            self.config.bpm_weight * profiling_output.anomaly_score
            + self.config.sdm_weight * detection_output.anomaly_score
        )
        return UnifiedAnomalySignal(
            entity_id=profiling_output.entity_id,
            event_id=profiling_output.event_id,
            fused_score=fused_score,
            is_anomaly=fused_score >= self.config.fusion_threshold,
            bpm_score=profiling_output.anomaly_score,
            sdm_score=detection_output.anomaly_score,
            cold_start_flag=(
                profiling_output.cold_start_flag or detection_output.cold_start_flag
            ),
            contributing_features=self._merge_contributing_features(
                profiling_output.top_contributing_features,
                detection_output.top_contributing_features,
            ),
        )

    @staticmethod
    def _validate_matching_ids(
        profiling_output: ProfilingOutput, detection_output: DetectionOutput
    ) -> None:
        """Reject accidental fusion of scores from different inference events."""
        if profiling_output.entity_id != detection_output.entity_id:
            raise ValueError("BPM and SDM outputs must have matching entity_id values")
        if profiling_output.event_id != detection_output.event_id:
            raise ValueError("BPM and SDM outputs must have matching event_id values")

    @staticmethod
    def _merge_contributing_features(
        bpm_features: list[str], sdm_features: list[str]
    ) -> list[str]:
        """Return the stable, deduplicated union required by boundary G."""
        return list(dict.fromkeys([*bpm_features, *sdm_features]))
