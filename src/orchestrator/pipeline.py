"""Dependency-injected end-to-end inference pipeline (ML pipeline M12)."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol
import asyncio

from anomaly_detection.common.models.access_log import AccessLogInference
from anomaly_detection.common.models.alerts import Alert
from anomaly_detection.common.models.entities import EntityProfile
from anomaly_detection.common.models.enums import EntityType
from anomaly_detection.common.models.features import EngineeredFeatures
from anomaly_detection.common.models.ml_io import (
    ClassificationOutput,
    DetectionOutput,
    Explanation,
    ProfilingOutput,
    UnifiedAnomalySignal,
)

from .alert_builder import AlertBuilder
from .config import OrchestratorConfig


class FeaturePipelineInterface(Protocol):
    """Boundary-C producer required by the orchestrator."""

    def transform_single(
        self, record: AccessLogInference, profile: EntityProfile | None = None
    ) -> EngineeredFeatures:
        """Produce one feature vector and sequence window for an inference event."""


class ProfilingModelInterface(Protocol):
    """Boundary-F BPM scorer required by the orchestrator."""

    def score(
        self,
        entity_id: str,
        event_id: str,
        entity_type: EntityType,
        feature_vector: object,
    ) -> ProfilingOutput:
        """Produce the profiling anomaly score for a feature vector."""


class DetectionModelInterface(Protocol):
    """Boundary-F SDM scorer required by the orchestrator."""

    def predict(self, features: EngineeredFeatures) -> DetectionOutput:
        """Produce the sequence anomaly score for engineered features."""


class ScoreFusionInterface(Protocol):
    """Boundary-G producer required by the orchestrator."""

    def fuse(
        self,
        profiling_output: ProfilingOutput,
        detection_output: DetectionOutput,
    ) -> UnifiedAnomalySignal:
        """Fuse matching model scores into a unified anomaly signal."""


class ClassifierInterface(Protocol):
    """Boundary-H producer required by the orchestrator."""

    def classify_signal(
        self, signal: UnifiedAnomalySignal, feature_vector: object
    ) -> ClassificationOutput:
        """Classify a fused signal using its engineered feature vector."""


class ExplainabilityInterface(Protocol):
    """Boundary-I explanation producer required by the orchestrator."""

    def explain(
        self,
        detection_output: DetectionOutput,
        classification_output: ClassificationOutput,
        profiling_output: ProfilingOutput,
        feature_vector: object,
        **kwargs: object,
    ) -> Explanation:
        """Generate analyst-facing evidence and a narrative."""


class ProfileStoreInterface(Protocol):
    """Minimal profile lookup used to avoid duplicate feature-pipeline reads."""

    def get_profile(self, entity_id: str) -> EntityProfile | None:
        """Return an entity profile when it exists."""


class ColdStartHandlerInterface(Protocol):
    """Cold-start graduation hook executed after BPM scoring."""

    def handle(
        self, entity_id: str, feature_vector: object, raw_profiling_output: ProfilingOutput
    ) -> ProfilingOutput:
        """Apply cold-start confidence policy and graduation tracking."""


class EWMAUpdaterInterface(Protocol):
    """Step-5b updater executed before an alert can be persisted."""

    def update(
        self, entity_id: str, anomaly_score: float, session_features: object
    ) -> EntityProfile | None:
        """Safely adapt an established profile when the score passes its gate."""


class AlertStoreInterface(Protocol):
    """Persistence boundary for completed alerts."""

    def save_alert(self, alert: Alert) -> None:
        """Persist a fully constructed alert."""


class InferencePipeline:
    """Execute the exact M12 flow for one label-free access-log event.

    Components are constructor-injected, so the pipeline has no hard-coded
    model or storage imports and can be tested at every boundary.  The order
    is fixed: features, BPM/cold-start, SDM, fusion, classification,
    explanation, gated EWMA, then alert construction and persistence.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        feature_pipeline: FeaturePipelineInterface,
        profile_store: ProfileStoreInterface,
        profiling_model: ProfilingModelInterface,
        detection_model: DetectionModelInterface,
        score_fusion: ScoreFusionInterface,
        classifier: ClassifierInterface,
        explainability: ExplainabilityInterface,
        ewma_updater: EWMAUpdaterInterface,
        alert_store: AlertStoreInterface,
        cold_start_handler: ColdStartHandlerInterface | None = None,
        alert_builder: AlertBuilder | None = None,
        alert_stream_queue: asyncio.Queue | None = None,
    ) -> None:
        """Initialize the pipeline with every upstream and downstream dependency."""
        self.config = config
        self.feature_pipeline = feature_pipeline
        self.profile_store = profile_store
        self.profiling_model = profiling_model
        self.detection_model = detection_model
        self.score_fusion = score_fusion
        self.classifier = classifier
        self.explainability = explainability
        self.ewma_updater = ewma_updater
        self.alert_store = alert_store
        self.cold_start_handler = cold_start_handler
        self.alert_builder = alert_builder or AlertBuilder()
        self.alert_stream_queue = alert_stream_queue
        self.last_processing_latency_ms = 0.0

    def process(self, event: AccessLogInference) -> Alert | None:
        """Process one inference event and persist an alert only above threshold."""
        started_at = perf_counter()
        profile = self.profile_store.get_profile(event.entity_id)
        features = self.feature_pipeline.transform_single(event, profile)
        profiling_output = self.profiling_model.score(
            event.entity_id,
            event.event_id,
            EntityType(event.entity_type),
            features.feature_vector,
        )
        if self.cold_start_handler is not None:
            profiling_output = self.cold_start_handler.handle(
                event.entity_id,
                features.feature_vector,
                profiling_output,
            )
        detection_output = self.detection_model.predict(features)
        signal = self.score_fusion.fuse(profiling_output, detection_output)
        classification = self.classifier.classify_signal(signal, features.feature_vector)
        explanation = self.explainability.explain(
            detection_output,
            classification,
            profiling_output,
            features.feature_vector,
            raw_snapshot=event.model_dump(mode="json", exclude={"delivery_mode"}),
        )

        # Step 5b must happen before any alert-store write, including normals.
        self.ewma_updater.update(event.entity_id, signal.fused_score, features.feature_vector.root)

        if signal.fused_score < self.config.alert_threshold:
            self._record_latency(started_at)
            return None

        alert = self.alert_builder.build(event, signal, classification, explanation)
        self.alert_store.save_alert(alert)
        
        if self.alert_stream_queue is not None:
            # We construct the lightweight summary to push
            from anomaly_detection.common.models.alerts import AlertSummary
            summary = AlertSummary(
                alert_id=alert.alert_id,
                entity_id=alert.entity_id,
                timestamp=alert.timestamp,
                risk_score=alert.risk.risk_score,
                risk_tier=alert.risk.risk_tier,
                attack_class=alert.attack_class,
                classification_confidence=alert.classification_confidence,
                cold_start_flag=alert.cold_start_flag,
                human_readable_explanation=alert.explanation.human_readable_explanation[:150]
            )
            # using put_nowait so it doesn't block the synchronous processing thread if possible
            # wait, this process() is synchronous, so we can use put_nowait?
            # Actually put_nowait is safe from synchronous threads if called from the same event loop.
            # But process() might be running in a thread pool (FastAPI BackgroundTasks run in a threadpool).
            # We must use call_soon_threadsafe!
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(self.alert_stream_queue.put_nowait, summary)
            except RuntimeError:
                pass
                
        self._record_latency(started_at)
        return alert

    def _record_latency(self, started_at: float) -> None:
        """Retain the latest observed processing duration for latency monitoring."""
        self.last_processing_latency_ms = (perf_counter() - started_at) * 1000.0
