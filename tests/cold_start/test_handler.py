import os
import pytest
from pydantic import Field

from src.cold_start.config import ColdStartConfig
from src.cold_start.handler import ColdStartHandler
from src.cold_start.event_counter import EventCounter
from src.profiling.profile_store import ProfileStore
from anomaly_detection.common.models.features import EntityFeatureVector
from anomaly_detection.common.models.ml_io import ProfilingOutput
from anomaly_detection.common.models.enums import EntityStatus, EntityType
from anomaly_detection.common.models.entities import EntityProfile


# Mock ExtendedProfilingOutput since it's defined in src/profiling/profile_model.py
class MockExtendedProfilingOutput(ProfilingOutput):
    entity_status: EntityStatus = Field(EntityStatus.WARM)
    is_cold_start: bool = Field(False)
    profile_age: int = Field(0)


class MockProfileStore(ProfileStore):
    def __init__(self):
        super().__init__("data/cold_start_test/store.json")
        self.upsert_count = 0
        self.last_upsert = None

    def upsert_profile(self, profile: EntityProfile) -> None:
        super().upsert_profile(profile)
        self.upsert_count += 1
        self.last_upsert = profile


@pytest.fixture
def config(tmp_path):
    counter_path = os.path.join(tmp_path, "event_counter.json")
    return ColdStartConfig(
        graduation_threshold=10,
        confidence_discount_factor=0.6,
        ambiguity_threshold=0.65,
        event_counter_persistence_path=counter_path
    )


@pytest.fixture
def store(tmp_path):
    store = MockProfileStore()
    store.persistence_path = os.path.join(tmp_path, "store.json")
    return store


@pytest.fixture
def handler(config, store):
    return ColdStartHandler(config, store)


def create_mock_output(entity_id="usr_1234abcd", confidence=1.0):
    return MockExtendedProfilingOutput(
        entity_id=entity_id,
        event_id="evt_123",
        model_id="bpm",
        anomaly_score=0.1,
        confidence=confidence,
        cold_start_flag=False,
        top_contributing_features=["f1"],
        entity_status=EntityStatus.WARM,
        is_cold_start=False,
        profile_age=0
    )


def test_cold_start_detection_and_discount(handler):
    entity_id = "usr_1234abcd"
    fvec = EntityFeatureVector(root=[0.1] * 24)
    raw_output = create_mock_output(entity_id, confidence=1.0)
    
    output = handler.handle(entity_id, fvec, raw_output)
    
    # 1. entity not in ProfileStore -> entity_status=cold_start, is_cold_start=True
    assert output.entity_status == EntityStatus.COLD_START
    assert output.is_cold_start is True
    
    # 2. ambiguity_reason populated
    assert hasattr(output, "ambiguity_reason")
    assert "no behavioral history available" in getattr(output, "ambiguity_reason")
    
    # 3. Confidence discount exact arithmetic
    assert output.confidence == 1.0 * 0.6


def test_graduation_rule(handler, store):
    entity_id = "svc_5678efgh"
    
    # 9 events -> shouldn't graduate
    for i in range(9):
        fvec = EntityFeatureVector(root=[0.1] * 24)
        raw = create_mock_output(entity_id, 1.0)
        out = handler.handle(entity_id, fvec, raw)
        assert out.entity_status == EntityStatus.COLD_START
        assert store.upsert_count == 0

    # 10th event -> graduates
    fvec = EntityFeatureVector(root=[0.1] * 24)
    raw = create_mock_output(entity_id, 1.0)
    out = handler.handle(entity_id, fvec, raw)
    
    # The 10th event itself is still scored as cold-start per the logic
    # (since it was passed to handle while cold-start)
    # But upsert should be called.
    assert store.upsert_count == 1
    
    # Verify the profile that was upserted
    assert store.last_upsert is not None
    assert store.last_upsert.entity_type == EntityType.SERVICE_ACCOUNT
    assert store.last_upsert.event_count == 10
    
    # 11th event -> warm entity passthrough
    fvec = EntityFeatureVector(root=[0.1] * 24)
    raw = create_mock_output(entity_id, 0.9)
    out = handler.handle(entity_id, fvec, raw)
    
    assert out.confidence == 0.9
    assert out.entity_status == EntityStatus.WARM
    assert out.is_cold_start is False


def test_post_graduation_passthrough(handler, store):
    entity_id = "dev_9999ffff"
    
    # Create a warm profile
    profile = EntityProfile(
        entity_id=entity_id,
        entity_type=EntityType.EDGE_DEVICE,
        baseline_vector=[0.0]*24,
        baseline_std=[1.0]*24,
        sequence_history=[],
        most_frequent_country="US",
        known_mac_addresses=[],
        known_os_profiles=[],
        known_protocols=[],
        resource_access_counts={},
        command_frequency={},
        event_count=15,  # >= graduation_threshold
        cold_start_flag=False,
        last_updated="2026-07-26T00:00:00Z",
        profile_version=1
    )
    store.upsert_profile(profile)
    
    fvec = EntityFeatureVector(root=[0.1] * 24)
    raw = create_mock_output(entity_id, 0.95)
    
    out = handler.handle(entity_id, fvec, raw)
    
    # Unchanged
    assert out.confidence == 0.95
    assert out.entity_status == EntityStatus.WARM
    assert out.is_cold_start is False


def test_event_counter_persistence(tmp_path):
    counter_path = os.path.join(tmp_path, "counter.json")
    counter1 = EventCounter(counter_path)
    
    counter1.increment("test_usr", [0.1]*24)
    counter1.increment("test_usr", [0.2]*24)
    assert counter1.get_count("test_usr") == 2
    
    # Simulate restart
    counter2 = EventCounter(counter_path)
    assert counter2.get_count("test_usr") == 2
    vectors = counter2.get_vectors("test_usr")
    assert len(vectors) == 2
    assert vectors[0] == [0.1]*24


def test_post_graduation_cleanup(handler, store):
    entity_id = "usr_cleanup"
    
    # Trigger graduation
    for i in range(10):
        fvec = EntityFeatureVector(root=[0.1] * 24)
        raw = create_mock_output(entity_id, 1.0)
        handler.handle(entity_id, fvec, raw)
        
    assert store.upsert_count == 1
    # Check cleanup
    assert handler.counter.get_count(entity_id) == 0
    assert len(handler.counter.get_vectors(entity_id)) == 0

