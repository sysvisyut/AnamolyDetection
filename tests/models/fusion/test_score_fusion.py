"""Tests for ``models.fusion.score_fusion``.

Tier: T1
Pytest mark: @pytest.mark.tier1
"""

import pytest
from anomaly_detection.common.models.ml_io import DetectionOutput, ProfilingOutput
from anomaly_detection.models.fusion import FusionConfig, ScoreFusion


def make_outputs() -> tuple[ProfilingOutput, DetectionOutput]:
    """Create matching boundary-F outputs with overlapping feature names."""
    return (
        ProfilingOutput(
            entity_id="usr_001",
            event_id="evt_001",
            anomaly_score=0.8,
            confidence=1.0,
            cold_start_flag=False,
            top_contributing_features=["geo_velocity", "failure_count"],
        ),
        DetectionOutput(
            entity_id="usr_001",
            event_id="evt_001",
            anomaly_score=0.4,
            confidence=1.0,
            cold_start_flag=True,
            top_contributing_features=["failure_count", "command_rarity"],
        ),
    )


def test_fuse_returns_continuous_weighted_signal() -> None:
    """Fusion preserves component scores and applies the configured threshold."""
    profiling_output, detection_output = make_outputs()

    signal = ScoreFusion().fuse(profiling_output, detection_output)

    assert signal.fused_score == pytest.approx(0.6)
    assert signal.is_anomaly is True
    assert signal.bpm_score == 0.8
    assert signal.sdm_score == 0.4
    assert signal.cold_start_flag is True
    assert signal.contributing_features == [
        "geo_velocity",
        "failure_count",
        "command_rarity",
    ]


def test_fuse_uses_injected_weights_and_threshold() -> None:
    """Tuning changes only the configured convex combination and decision gate."""
    profiling_output, detection_output = make_outputs()
    fusion = ScoreFusion(
        FusionConfig(bpm_weight=0.25, sdm_weight=0.75, fusion_threshold=0.51)
    )

    signal = fusion.fuse(profiling_output, detection_output)

    assert signal.fused_score == pytest.approx(0.5)
    assert signal.is_anomaly is False


def test_fuse_rejects_scores_for_different_events() -> None:
    """Fusion cannot accidentally join unrelated entity observations."""
    profiling_output, detection_output = make_outputs()
    detection_output.event_id = "evt_other"

    with pytest.raises(ValueError, match="event_id"):
        ScoreFusion().fuse(profiling_output, detection_output)


def test_fusion_config_requires_weights_to_sum_to_one() -> None:
    """Invalid score weighting is rejected before any inference occurs."""
    with pytest.raises(ValueError, match="sum"):
        FusionConfig(bpm_weight=0.6, sdm_weight=0.6)
