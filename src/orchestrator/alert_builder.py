"""Construction of analyst-facing alerts from completed inference outputs."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from anomaly_detection.common.models.access_log import AccessLogInference
from anomaly_detection.common.models.alerts import (
    Alert,
    RiskScore,
)
from anomaly_detection.common.models.alerts import (
    Explanation as AlertExplanation,
)
from anomaly_detection.common.models.alerts import (
    FeatureAttribution as AlertFeatureAttribution,
)
from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.common.models.ml_io import (
    ClassificationOutput,
    UnifiedAnomalySignal,
)
from anomaly_detection.common.models.ml_io import (
    Explanation as ModelExplanation,
)

MAX_INSIDER_DRIFT_RISK_SCORE = 49
RISK_SCORE_MULTIPLIER = 100


class AlertBuilder:
    """Map typed model outputs into the boundary-I ``Alert`` contract."""

    def build(
        self,
        event: AccessLogInference,
        signal: UnifiedAnomalySignal,
        classification: ClassificationOutput,
        explanation: ModelExplanation,
    ) -> Alert:
        """Build a complete alert without retaining inference-only metadata."""
        risk_score = self._risk_score(signal.fused_score, classification.predicted_class)
        return Alert(
            alert_id=str(uuid4()),
            entity_id=event.entity_id,
            event_id=event.event_id,
            session_id=event.session_id,
            timestamp=event.timestamp,
            detected_at=datetime.now(UTC).isoformat(),
            risk=RiskScore(
                risk_score=risk_score,
                risk_tier=self._risk_tier(risk_score),
            ),
            explanation=AlertExplanation(
                human_readable_explanation=explanation.narrative,
                feature_attributions=[
                    AlertFeatureAttribution.model_validate(attribution)
                    for attribution in explanation.feature_attributions
                ],
            ),
            attack_class=classification.predicted_class,
            classification_confidence=classification.classification_confidence,
            fused_score=signal.fused_score,
            bpm_score=signal.bpm_score,
            sdm_score=signal.sdm_score,
            cold_start_flag=signal.cold_start_flag,
            raw_event_snapshot=event.model_dump(mode="json", exclude={"delivery_mode"}),
        )

    @staticmethod
    def _risk_score(fused_score: float, category: AnomalyCategory) -> int:
        """Convert the normalized fused score to the configured 0–100 scale."""
        risk_score = round(fused_score * RISK_SCORE_MULTIPLIER)
        if category == AnomalyCategory.INSIDER_DRIFT:
            return min(risk_score, MAX_INSIDER_DRIFT_RISK_SCORE)
        return risk_score

    @staticmethod
    def _risk_tier(risk_score: int) -> str:
        """Assign the DATA_SCHEMA.md severity band for a 0–100 risk score."""
        if risk_score <= 24:
            return "low"
        if risk_score <= 49:
            return "medium"
        if risk_score <= 74:
            return "high"
        return "critical"
