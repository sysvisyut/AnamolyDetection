"""
Tests for shared data models (M02).

@pytest.mark.tier1
"""

import pytest
from pydantic import ValidationError

from anomaly_detection.common.models import (
    MODULE_SCHEMA_VERSION,
    EntityType,
    AuthMethod,
    AnomalyCategory,
    EntityStatus,
    GeoLocation,
    CommandEntry,
    DeviceFingerprint,
    AccessLogTraining,
    AccessLogInference,
    SessionMetadata,
    EntityFeatureVector,
    EntitySequence,
    EngineeredFeatures,
    ProfilingOutput,
    DetectionOutput,
    UnifiedAnomalySignal,
    ClassificationOutput,
    FeatureAttribution,
    RiskScore,
    Explanation,
    Alert,
    DriftMetrics,
    EntityProfile,
    EntityHistoryEntry,
)


@pytest.mark.tier1
def test_schema_version() -> None:
    """MODULE_SCHEMA_VERSION matches DATA_SCHEMA.md's declared version."""
    assert MODULE_SCHEMA_VERSION == "1.0"


@pytest.mark.tier1
def test_enum_completeness() -> None:
    """confirm AnomalyCategory contains all 7 attack types + insider_drift + unclassified."""
    expected_categories = {
        "normal",
        "brute_force",
        "impossible_travel",
        "credential_stuffing",
        "lateral_movement",
        "device_spoofing",
        "low_and_slow",
        "insider_drift",
        "unclassified",
    }
    actual_categories = {cat.value for cat in AnomalyCategory}
    assert actual_categories == expected_categories


@pytest.fixture
def valid_geo_location() -> dict:
    return {
        "city": "Mumbai",
        "country": "IN",
        "latitude": 19.0760,
        "longitude": 72.8777,
    }


@pytest.fixture
def valid_device_fingerprint() -> dict:
    return {
        "device_id": "dev_123",
        "os_family": "Windows",
        "os_version": "11.0",
        "mac_address": "AA:BB:CC:DD:EE:FF",
        "protocol": "HTTPS",
        "user_agent": "Mozilla/5.0",
        "firmware_version": "",
    }


@pytest.fixture
def valid_command_sequence() -> list:
    return [
        {
            "sequence_position": 0,
            "command": "ls",
            "target": "/etc",
            "outcome": "success",
            "elapsed_seconds": 2.5,
        }
    ]


@pytest.fixture
def valid_access_log_base(valid_geo_location, valid_device_fingerprint, valid_command_sequence) -> dict:
    return {
        "event_id": "uuid-1",
        "session_id": "sess-1",
        "entity_id": "usr-1",
        "entity_type": "user",
        "timestamp": "2026-07-25T10:00:00Z",
        "source_ip": "192.168.1.1",
        "geo_location": valid_geo_location,
        "resource_accessed": "file/secret.txt",
        "auth_method": "password",
        "auth_outcome": "success",
        "session_duration": 120.5,
        "command_sequence": valid_command_sequence,
        "device_fingerprint": valid_device_fingerprint,
        "failure_count": 0,
    }


@pytest.mark.tier1
def test_access_log_training_valid(valid_access_log_base) -> None:
    data = dict(valid_access_log_base)
    data["label"] = "normal"
    log = AccessLogTraining(**data)
    assert log.label == AnomalyCategory.NORMAL
    
    # round-trip
    dumped = log.model_dump()
    re_log = AccessLogTraining(**dumped)
    assert log == re_log


@pytest.mark.tier1
def test_access_log_inference_separation(valid_access_log_base) -> None:
    data = dict(valid_access_log_base)
    data["delivery_mode"] = "batch"
    
    # Valid instantiation
    log = AccessLogInference(**data)
    assert log.delivery_mode == "batch"
    
    # Cannot be constructed with a label field
    data["label"] = "normal"
    with pytest.raises(ValidationError):
        AccessLogInference(**data)
    
    # Structurally absent
    assert not hasattr(log, "label")


@pytest.mark.tier1
def test_access_log_invalid_values(valid_access_log_base) -> None:
    data = dict(valid_access_log_base)
    data["label"] = "normal"
    
    # Out of range geo
    data["geo_location"]["latitude"] = 100.0
    with pytest.raises(ValidationError):
        AccessLogTraining(**data)
        
    data["geo_location"]["latitude"] = 19.0
    data["session_duration"] = -5.0
    with pytest.raises(ValidationError):
        AccessLogTraining(**data)


@pytest.mark.tier1
def test_features_models() -> None:
    v = EntityFeatureVector([0.5] * 24)
    assert len(v.root) == 24
    
    with pytest.raises(ValidationError):
        EntityFeatureVector([0.5] * 23)
        
    s = EntitySequence([[0.5] * 24, [0.1] * 24])
    assert len(s.root) == 2
    
    metadata = SessionMetadata(is_cold_start=False, delivery_mode_hint="batch", profile_event_count=50)
    
    ef = EngineeredFeatures(
        entity_id="usr-1",
        event_id="evt-1",
        session_id="sess-1",
        feature_vector=v,
        sequence_window=s,
        session_metadata=metadata,
    )
    
    assert ef.entity_id == "usr-1"


@pytest.mark.tier1
def test_ml_io_models() -> None:
    bpm = ProfilingOutput(
        entity_id="usr-1",
        event_id="evt-1",
        anomaly_score=0.8,
        confidence=0.9,
        cold_start_flag=False,
        top_contributing_features=["feat1"],
    )
    assert bpm.model_id == "bpm"
    
    with pytest.raises(ValidationError):
        # score > 1.0
        ProfilingOutput(
            entity_id="usr-1",
            event_id="evt-1",
            anomaly_score=1.5,
            confidence=0.9,
            cold_start_flag=False,
            top_contributing_features=["feat1"],
        )


@pytest.mark.tier1
def test_alert_model() -> None:
    attr = FeatureAttribution(
        feature_name="feat1",
        feature_value=0.5,
        attribution_score=0.9,
        direction="toward_anomaly",
        source_model="bpm",
        human_label="Feature 1",
    )
    
    risk = RiskScore(risk_score=95, risk_tier="critical")
    expl = Explanation(human_readable_explanation="Test explanation", feature_attributions=[attr])
    
    alert = Alert(
        alert_id="a-1",
        entity_id="usr-1",
        event_id="evt-1",
        session_id="sess-1",
        timestamp="time",
        detected_at="time",
        risk=risk,
        explanation=expl,
        attack_class="normal",
        classification_confidence=0.9,
        fused_score=0.8,
        bpm_score=0.7,
        sdm_score=0.9,
        cold_start_flag=False,
        raw_event_snapshot={"some": "data"},
    )
    
    assert alert.risk.risk_score == 95
    assert len(alert.explanation.feature_attributions) == 1
    
    dumped = alert.model_dump()
    re_alert = Alert(**dumped)
    assert alert == re_alert


@pytest.mark.tier1
def test_entity_profile() -> None:
    profile = EntityProfile(
        entity_id="usr-1",
        entity_type="user",
        baseline_vector=[0.0] * 24,
        baseline_std=[1.0] * 24,
        sequence_history=[[0.0] * 24],
        most_frequent_country="IN",
        known_mac_addresses=["AA:BB:CC"],
        known_os_profiles=[{"os_family": "Windows", "os_version": "10"}],
        known_protocols=["HTTP"],
        resource_access_counts={"res": 1},
        command_frequency={"cmd": 1},
        event_count=100,
        cold_start_flag=False,
        last_updated="2026-07-25",
        profile_version=1,
    )
    assert profile.entity_type == EntityType.USER

