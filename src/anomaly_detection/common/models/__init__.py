"""
Shared data schema models.

Matches DATA_SCHEMA.md v1.0 constraints.
"""

# Schema version checked at import time
MODULE_SCHEMA_VERSION = "1.0"

from anomaly_detection.common.models.enums import (
    EntityType,
    AuthMethod,
    AnomalyCategory,
    EntityStatus,
)

from anomaly_detection.common.models.access_log import (
    GeoLocation,
    CommandEntry,
    DeviceFingerprint,
    AccessLogBase,
    AccessLogTraining,
    AccessLogInference,
)

from anomaly_detection.common.models.features import (
    SessionMetadata,
    EntityFeatureVector,
    EntitySequence,
    EngineeredFeatures,
)

from anomaly_detection.common.models.ml_io import (
    ModelScore,
    ProfilingOutput,
    DetectionOutput,
    UnifiedAnomalySignal,
    ClassificationOutput,
)

from anomaly_detection.common.models.alerts import (
    FeatureAttribution,
    RiskScore,
    Explanation,
    Alert,
)

from anomaly_detection.common.models.entities import (
    DriftMetrics,
    EntityProfile,
    EntityHistoryEntry,
)

__all__ = [
    "MODULE_SCHEMA_VERSION",
    "EntityType",
    "AuthMethod",
    "AnomalyCategory",
    "EntityStatus",
    "GeoLocation",
    "CommandEntry",
    "DeviceFingerprint",
    "AccessLogBase",
    "AccessLogTraining",
    "AccessLogInference",
    "SessionMetadata",
    "EntityFeatureVector",
    "EntitySequence",
    "EngineeredFeatures",
    "ModelScore",
    "ProfilingOutput",
    "DetectionOutput",
    "UnifiedAnomalySignal",
    "ClassificationOutput",
    "FeatureAttribution",
    "RiskScore",
    "Explanation",
    "Alert",
    "DriftMetrics",
    "EntityProfile",
    "EntityHistoryEntry",
]
