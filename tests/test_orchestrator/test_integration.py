"""Integration tests for the orchestrator, fusion, classifier, and explainability layer.

Tier: T1
Pytest mark: @pytest.mark.tier1
"""

import pytest
import numpy as np
from pydantic import BaseModel

from anomaly_detection.common.models.access_log import AccessLogInference, GeoLocation, DeviceFingerprint
from anomaly_detection.common.models.features import EngineeredFeatures, EntityFeatureVector, EntitySequence, SessionMetadata
from anomaly_detection.common.models.ml_io import ProfilingOutput, DetectionOutput
from anomaly_detection.models.fusion import ScoreFusion, FusionConfig
from src.classification.classifier import AnomalyClassifier
from src.classification.config import ClassifierConfig
from src.explainability.engine import ExplainabilityEngine
from src.explainability.config import ExplainabilityConfig
from src.orchestrator import InferencePipeline, OrchestratorConfig
from src.orchestrator.alert_builder import AlertBuilder

from anomaly_detection.stores.alert_store import AbstractAlertStore

class DummyEWMAUpdater:
    def update(self, entity_id: str, score: float, feature_vector: list[float]) -> None:
        pass

class DummyProfileStore:
    def get_profile(self, entity_id: str):
        return None

class DummyFeaturePipeline:
    def transform_single(self, record, profile) -> EngineeredFeatures:
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

class DummyBPM:
    def score(self, *args) -> ProfilingOutput:
        return ProfilingOutput(
            entity_id="usr_001",
            event_id="evt_001",
            anomaly_score=0.9,
            confidence=0.9,
            cold_start_flag=False,
            top_contributing_features=["failure_count_norm"],
        )

class DummySDM:
    def predict(self, *args) -> DetectionOutput:
        return DetectionOutput(
            entity_id="usr_001",
            event_id="evt_001",
            anomaly_score=0.7,
            confidence=0.8,
            cold_start_flag=False,
            top_contributing_features=["command_rarity_score"],
        )

class DummyAlertStore(AbstractAlertStore):
    def __init__(self):
        self.alerts = []
    def save_alert(self, alert):
        self.alerts.append(alert)
    def get_alert(self, alert_id):
        pass
    def get_alerts(self, *args, **kwargs):
        pass
    def update_feedback(self, alert_id, decision, notes):
        pass
    def get_entity_history(self, entity_id, limit):
        pass
    def save_history_entry(self, entry, entity_id):
        pass

class DummyExplainability:
    def explain(self, detection_output, classification_output, profiling_output, feature_vector, **kwargs):
        from anomaly_detection.common.models.ml_io import Explanation, FeatureAttribution
        from anomaly_detection.common.models.enums import AnomalyCategory
        return Explanation(
            narrative="Dummy narrative",
            feature_attributions=[
                FeatureAttribution(
                    feature_name="dummy_feature",
                    feature_value=1.0,
                    attribution_score=0.9,
                    direction="toward_anomaly",
                    source_model="bpm",
                    human_label="Dummy"
                )
            ],
            predicted_category=AnomalyCategory.UNCLASSIFIED,
            consistency_check_passed=True,
            is_ambiguous=False,
            ambiguity_reason=None
        )


@pytest.fixture
def classifier_fixture():
    # Setup real classifier with a dummy lightgbm model for testing
    import lightgbm as lgb
    X = np.random.rand(10, 27)
    y = np.random.randint(0, 8, size=(10,))
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train({"objective": "multiclass", "num_class": 8, "verbosity": -1}, dtrain, num_boost_round=1)
    
    config = ClassifierConfig(confidence_threshold=0.0)
    clf = AnomalyClassifier(config)
    clf.model = model
    return clf


def test_orchestrator_integration_real_fusion(classifier_fixture):
    # Setup Fusion
    fusion = ScoreFusion(FusionConfig(bpm_weight=0.5, sdm_weight=0.5, fusion_threshold=0.5))
    
    # Setup Explainability
    explainability = DummyExplainability()
    
    # Setup Orchestrator
    config = OrchestratorConfig(alert_threshold=0.5)
    alert_store = DummyAlertStore()
    
    pipeline = InferencePipeline(
        config=config,
        feature_pipeline=DummyFeaturePipeline(),
        profile_store=DummyProfileStore(),
        profiling_model=DummyBPM(),
        detection_model=DummySDM(),
        score_fusion=fusion,
        classifier=classifier_fixture,
        explainability=explainability,
        ewma_updater=DummyEWMAUpdater(),
        alert_store=alert_store,
        alert_builder=AlertBuilder()
    )
    
    # Process
    event = AccessLogInference(
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
    
    alert = pipeline.process(event)
    
    assert alert is not None
    assert len(alert_store.alerts) == 1
    assert alert.alert_id == alert_store.alerts[0].alert_id
    
    # Verify fused score (0.9 * 0.5 + 0.7 * 0.5 = 0.8)
    # The output from dummy BPM is 0.9, SDM is 0.7. Fused = 0.8.
    assert abs(alert.risk.risk_score - 80) <= 1
    
    # Features from fusion should be passed through
    assert "dummy_feature" in [fa.feature_name for fa in alert.explanation.feature_attributions]
