"""
Feature Engineering package for the AI-Powered Behavioral Anomaly Detection system.

ARCHITECTURE COMPONENT: Feature Engineering
BOUNDARY RESPONSIBILITY: Consumes B (AccessLogBase); produces C (EngineeredFeatures).
TIER: T1

This package implements M06 of the development roadmap. It transforms raw
access-log records (Boundary B) into the 24-dimensional feature vectors and
20-step sequence windows (Boundary C) consumed by the BPM and SDM.

Public API:
    FeatureEngineeringConfig  — pipeline parameters and population fallbacks
    ProfileStoreInterface     — Protocol that M05 (stores/) must implement
    AbstractProfileStore      — Optional ABC for M05 implementors
    SessionBuilder            — groups events by entity/session, tracks state
    SequenceBuilder           — constructs sliding sequence windows
    FeatureExtractor          — computes 24-dim feature vectors
    FeaturePipeline           — end-to-end orchestrator (batch + inference)
    EventContext              — context object from SessionBuilder to FeatureExtractor

Dependency direction:
    feature_engineering → common, stores (ProfileStoreInterface only)
    feature_engineering must NOT import from: models/, classifier/, explainability/, api/
"""

from anomaly_detection.feature_engineering.config import (
    AUTH_METHOD_MAP,
    AUTH_OUTCOME_MAP,
    EXFIL_COMMANDS,
    FEATURE_DIM,
    MIN_PROFILE_EVENTS,
    DEFAULT_SEQUENCE_WINDOW,
    DEFAULT_STRIDE,
    FeatureEngineeringConfig,
)
from anomaly_detection.feature_engineering.profile_store_interface import (
    AbstractProfileStore,
    ProfileStoreInterface,
)
from anomaly_detection.feature_engineering.session_builder import (
    EventContext,
    SessionBuilder,
)
from anomaly_detection.feature_engineering.sequence_builder import SequenceBuilder
from anomaly_detection.feature_engineering.feature_extractor import FeatureExtractor
from anomaly_detection.feature_engineering.feature_pipeline import FeaturePipeline

__all__ = [
    # Config
    "FeatureEngineeringConfig",
    "FEATURE_DIM",
    "MIN_PROFILE_EVENTS",
    "DEFAULT_SEQUENCE_WINDOW",
    "DEFAULT_STRIDE",
    "EXFIL_COMMANDS",
    "AUTH_METHOD_MAP",
    "AUTH_OUTCOME_MAP",
    # Interface
    "ProfileStoreInterface",
    "AbstractProfileStore",
    # Builders & extractors
    "SessionBuilder",
    "EventContext",
    "SequenceBuilder",
    "FeatureExtractor",
    # Pipeline
    "FeaturePipeline",
]
