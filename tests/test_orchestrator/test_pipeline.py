"""Tests for the dependency-injected M12 inference pipeline.

Tier: T1
Pytest mark: @pytest.mark.tier1
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anomaly_detection.common.models.access_log import (
    AccessLogInference,
    DeviceFingerprint,
    GeoLocation,
)
from anomaly_detection.common.models.alerts import Alert
from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.common.models.features import (
    EngineeredFeatures,
    EntityFeatureVector,
    EntitySequence,
    SessionMetadata,
)
from anomaly_detection.common.models.ml_io import (
    ClassificationOutput,
    DetectionOutput,
    Explanation,
    FeatureAttribution,
    ProfilingOutput,
    UnifiedAnomalySignal,
)

from src.orchestrator import InferencePipeline, OrchestratorConfig


def make_event() -> AccessLogInference:
    """Create one valid label-free inference event."""
    return AccessLogInference(
        event_id="evt_001",
        session_id="ses_001",
        entity_id="usr_001",
        entity_type="user",
        timestamp="2026-07-26T12:00:00+00:00",
        source_ip="203.0.113.10",
        geo_location=GeoLocation(city="Mumbai", country="IN", latitude=19.076, longitude=72.8777),
        resource_accessed="file/reports/q1.xlsx",
        auth_method="password",
        auth_outcome="success",
        session_duration=120.0,
        command_sequence=[],
        device_fingerprint=DeviceFingerprint(
            device_id="dev_001",
            os_family="Linux",
            os_version="22.04",
            mac_address="AA:BB:CC:DD:EE:FF",
            protocol="HTTPS",
            user_agent="test-agent",
            firmware_version="",
        ),
        failure_count=0,
        delivery_mode="batch",
    )


@dataclass
class PipelineDoubles:
    """Small boundary doubles that record the orchestrator's call order."""

    fused_score: float
    call_order: list[str] = field(default_factory=list)
    stored_alerts: list[Alert] = field(default_factory=list)

    def get_profile(self, entity_id: str) -> None:
        """Return no profile to exercise the cold-start-safe path."""
        self.call_order.append("profile_lookup")
        return None

    def transform_single(
        self, record: AccessLogInference, profile: object = None
    ) -> EngineeredFeatures:
        """Produce deterministic boundary-C features."""
        self.call_order.append("features")
        return EngineeredFeatures(
            entity_id=record.entity_id,
            event_id=record.event_id,
            session_id=record.session_id,
            feature_vector=EntityFeatureVector(root=[0.1] * 24),
            sequence_window=EntitySequence(root=[[0.1] * 24]),
            session_metadata=SessionMetadata(
                is_cold_start=True,
                delivery_mode_hint=record.delivery_mode,
                profile_event_count=0,
            ),
        )

    def score(self, *args: object) -> ProfilingOutput:
        """Return a deterministic BPM boundary-F output."""
        self.call_order.append("bpm")
        return ProfilingOutput(
            entity_id="usr_001",
            event_id="evt_001",
            anomaly_score=self.fused_score,
            confidence=0.6,
            cold_start_flag=True,
            top_contributing_features=["failure_count_norm"],
        )

    def handle(
        self, entity_id: str, feature_vector: object, raw_profiling_output: ProfilingOutput
    ) -> ProfilingOutput:
        """Record cold-start handling without changing the supplied output."""
        self.call_order.append("cold_start")
        return raw_profiling_output

    def predict(self, features: EngineeredFeatures) -> DetectionOutput:
        """Return a deterministic SDM boundary-F output."""
        self.call_order.append("sdm")
        return DetectionOutput(
            entity_id=features.entity_id,
            event_id=features.event_id,
            anomaly_score=self.fused_score,
            confidence=0.5,
            cold_start_flag=True,
            top_contributing_features=["command_rarity_score"],
        )

    def fuse(
        self, profiling_output: ProfilingOutput, detection_output: DetectionOutput
    ) -> UnifiedAnomalySignal:
        """Produce the fused signal consumed by the classifier."""
        self.call_order.append("fusion")
        return UnifiedAnomalySignal(
            entity_id=profiling_output.entity_id,
            event_id=profiling_output.event_id,
            fused_score=self.fused_score,
            is_anomaly=self.fused_score >= 0.5,
            bpm_score=profiling_output.anomaly_score,
            sdm_score=detection_output.anomaly_score,
            cold_start_flag=True,
            contributing_features=["failure_count_norm", "command_rarity_score"],
        )

    def classify_signal(
        self, signal: UnifiedAnomalySignal, feature_vector: object
    ) -> ClassificationOutput:
        """Produce a high-confidence attack classification."""
        self.call_order.append("classification")
        return ClassificationOutput(
            entity_id=signal.entity_id,
            event_id=signal.event_id,
            predicted_class=AnomalyCategory.BRUTE_FORCE,
            class_probabilities={AnomalyCategory.BRUTE_FORCE.value: 1.0},
            classification_confidence=1.0,
            is_anomaly=signal.is_anomaly,
        )

    def explain(self, *args: object, **kwargs: object) -> Explanation:
        """Produce a valid explainability boundary with one attribution."""
        self.call_order.append("explanation")
        return Explanation(
            narrative="Brute force attack detected due to repeated failures.",
            feature_attributions=[
                FeatureAttribution(
                    feature_name="failure_count_norm",
                    feature_value=0.9,
                    attribution_score=0.9,
                    direction="toward_anomaly",
                    source_model="bpm",
                    human_label="Authentication failures",
                )
            ],
            predicted_category=AnomalyCategory.BRUTE_FORCE,
            consistency_check_passed=True,
            is_ambiguous=False,
        )

    def update(
        self, entity_id: str, anomaly_score: float, session_features: object
    ) -> None:
        """Record Step 5b without changing a mock profile."""
        self.call_order.append("ewma")

    def save_alert(self, alert: Alert) -> None:
        """Record an alert-store persistence operation."""
        self.call_order.append("persist")
        self.stored_alerts.append(alert)


def make_pipeline(doubles: PipelineDoubles) -> InferencePipeline:
    """Wire one set of doubles into the production dependency-injection seam."""
    return InferencePipeline(
        config=OrchestratorConfig(),
        feature_pipeline=doubles,
        profile_store=doubles,
        profiling_model=doubles,
        detection_model=doubles,
        score_fusion=doubles,
        classifier=doubles,
        explainability=doubles,
        ewma_updater=doubles,
        alert_store=doubles,
        cold_start_handler=doubles,
    )


def test_process_persists_alert_after_gated_ewma_step() -> None:
    """A score at the alert threshold returns a complete persisted alert."""
    doubles = PipelineDoubles(fused_score=0.8)
    pipeline = make_pipeline(doubles)

    alert = pipeline.process(make_event())

    assert alert is not None
    assert alert.attack_class == AnomalyCategory.BRUTE_FORCE
    assert alert.risk.risk_score == 80
    assert alert.raw_event_snapshot["event_id"] == "evt_001"
    assert "delivery_mode" not in alert.raw_event_snapshot
    assert doubles.stored_alerts == [alert]
    assert doubles.call_order == [
        "profile_lookup",
        "features",
        "bpm",
        "cold_start",
        "sdm",
        "fusion",
        "classification",
        "explanation",
        "ewma",
        "persist",
    ]
    assert pipeline.last_processing_latency_ms <= pipeline.config.max_processing_latency_ms


def test_process_returns_none_for_normal_but_still_updates_profile() -> None:
    """Below-threshold events are not persisted but complete required Step 5b."""
    doubles = PipelineDoubles(fused_score=0.2)
    pipeline = make_pipeline(doubles)

    alert = pipeline.process(make_event())

    assert alert is None
    assert doubles.stored_alerts == []
    assert doubles.call_order[-1] == "ewma"
