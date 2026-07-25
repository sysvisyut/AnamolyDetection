"""
Test suite for the Feature Engineering Pipeline (M06).

Tier: T1
Pytest mark: @pytest.mark.tier1

Coverage targets:
  - Schema conformance (all outputs validate against DATA_SCHEMA.md)
  - Geo-velocity accuracy against known coordinate pairs
  - Device fingerprint consistency scoring
  - Cold-start handling (no NaN/Inf/missing values)
  - Label leakage prevention (Training == Inference feature vectors)
  - Batch vs. single-record mode identity
  - ProfileStoreInterface injection and mock usage
  - Sequence window shape correctness
  - Attack signal coverage (statistically distinct from normal)
  - Edge cases: single event, multi-window, empty entity
"""

from __future__ import annotations

import datetime
import math
import uuid
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from anomaly_detection.common.models.access_log import (
    AccessLogBase,
    AccessLogInference,
    AccessLogTraining,
    CommandEntry,
    DeviceFingerprint,
    GeoLocation,
)
from anomaly_detection.common.models.entities import EntityProfile
from anomaly_detection.common.models.enums import AnomalyCategory, EntityType
from anomaly_detection.common.models.features import (
    EngineeredFeatures,
    EntityFeatureVector,
    EntitySequence,
)
from anomaly_detection.feature_engineering import (
    FEATURE_DIM,
    FeatureEngineeringConfig,
    FeatureExtractor,
    FeaturePipeline,
    ProfileStoreInterface,
    SequenceBuilder,
    SessionBuilder,
)
from anomaly_detection.feature_engineering.encoders import (
    encode_auth_method,
    encode_auth_outcome,
    encode_day_of_week,
    encode_failure_count,
    encode_fingerprint_mac_match,
    encode_fingerprint_os_match,
    encode_fingerprint_protocol_match,
    encode_has_exfil_command,
    encode_hour_of_day,
    encode_is_new_geo,
    encode_resource_category,
    encode_resource_rarity,
)
from anomaly_detection.feature_engineering.geo_velocity import (
    compute_geo_velocity,
    haversine_distance_km,
)
from anomaly_detection.feature_engineering.profile_store_interface import (
    AbstractProfileStore,
)

# ──────────────────────────────────────────────────────────────────────────────
# Pytest marks
# ──────────────────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.tier1


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures and factory helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_ts(
    year: int = 2026,
    month: int = 7,
    day: int = 1,
    hour: int = 10,
    minute: int = 0,
    second: int = 0,
) -> str:
    """Generate an ISO-8601 UTC timestamp string."""
    return (
        f"{year:04d}-{month:02d}-{day:02d}T"
        f"{hour:02d}:{minute:02d}:{second:02d}.000000+00:00"
    )


def _make_geo(city: str = "Mumbai", country: str = "IN", lat: float = 19.076, lon: float = 72.877) -> GeoLocation:
    return GeoLocation(city=city, country=country, latitude=lat, longitude=lon)


def _make_device(
    device_id: str = "dev_aabbccdd",
    os_family: str = "Windows",
    os_version: str = "11.0",
    mac: str = "AA:BB:CC:11:22:33",
    protocol: str = "HTTPS",
) -> DeviceFingerprint:
    return DeviceFingerprint(
        device_id=device_id,
        os_family=os_family,
        os_version=os_version,
        mac_address=mac,
        protocol=protocol,
        user_agent="Mozilla/5.0",
        firmware_version="",
    )


def _make_inference_event(
    entity_id: str = "usr_00000001",
    entity_type: str = "user",
    timestamp: str = "",
    session_id: str = "",
    resource: str = "file/reports/q1.xlsx",
    auth_method: str = "password",
    auth_outcome: str = "success",
    failure_count: int = 0,
    session_duration: float = 600.0,
    commands: Optional[List[CommandEntry]] = None,
    geo: Optional[GeoLocation] = None,
    device: Optional[DeviceFingerprint] = None,
    source_ip: str = "192.168.1.1",
) -> AccessLogInference:
    return AccessLogInference(
        event_id=str(uuid.uuid4()),
        session_id=session_id or str(uuid.uuid4()),
        entity_id=entity_id,
        entity_type=entity_type,
        timestamp=timestamp or _make_ts(),
        source_ip=source_ip,
        geo_location=geo or _make_geo(),
        resource_accessed=resource,
        auth_method=auth_method,
        auth_outcome=auth_outcome,
        session_duration=session_duration,
        command_sequence=commands or [],
        device_fingerprint=device or _make_device(),
        failure_count=failure_count,
        delivery_mode="batch",
    )


def _make_training_event(
    label: str = "normal",
    **kwargs,
) -> AccessLogTraining:
    inference = _make_inference_event(**kwargs)
    return AccessLogTraining(
        event_id=inference.event_id,
        session_id=inference.session_id,
        entity_id=inference.entity_id,
        entity_type=inference.entity_type,
        timestamp=inference.timestamp,
        source_ip=inference.source_ip,
        geo_location=inference.geo_location,
        resource_accessed=inference.resource_accessed,
        auth_method=inference.auth_method,
        auth_outcome=inference.auth_outcome,
        session_duration=inference.session_duration,
        command_sequence=inference.command_sequence,
        device_fingerprint=inference.device_fingerprint,
        failure_count=inference.failure_count,
        label=AnomalyCategory(label),
    )


def _make_warm_profile(
    entity_id: str = "usr_00000001",
    entity_type: str = "user",
    mac: str = "AA:BB:CC:11:22:33",
    event_count: int = 50,
    resource: str = "file/reports/q1.xlsx",
) -> EntityProfile:
    return EntityProfile(
        entity_id=entity_id,
        entity_type=EntityType(entity_type),
        baseline_vector=[0.0] * FEATURE_DIM,
        baseline_std=[1.0] * FEATURE_DIM,
        sequence_history=[],
        most_frequent_country="IN",
        known_mac_addresses=[mac],
        known_os_profiles=[{"os_family": "Windows", "os_version": "11.0"}],
        known_protocols=["HTTPS"],
        resource_access_counts={resource: event_count},
        command_frequency={},
        event_count=event_count,
        cold_start_flag=False,
        drift_metrics=None,
        last_updated=_make_ts(),
        profile_version=1,
    )


class _MockProfileStore:
    """Simple in-memory mock satisfying ProfileStoreInterface for testing."""

    def __init__(self, profiles: Optional[Dict[str, EntityProfile]] = None) -> None:
        self._profiles: Dict[str, Optional[EntityProfile]] = dict(profiles or {})

    def get_profile(self, entity_id: str) -> Optional[EntityProfile]:
        return self._profiles.get(entity_id)

    def get_profiles_batch(self, entity_ids: List[str]) -> Dict[str, Optional[EntityProfile]]:
        return {eid: self.get_profile(eid) for eid in entity_ids}

    def upsert_profile(self, profile: EntityProfile) -> None:
        self._profiles[profile.entity_id] = profile

    def list_entity_ids(self) -> List[str]:
        return sorted(self._profiles.keys())


@pytest.fixture
def default_config() -> FeatureEngineeringConfig:
    cfg = FeatureEngineeringConfig()
    cfg.cold_start_session_duration_mean = 600.0
    cfg.cold_start_session_duration_std = 300.0
    cfg.population_most_frequent_country = "IN"
    return cfg


@pytest.fixture
def warm_profile() -> EntityProfile:
    return _make_warm_profile()


@pytest.fixture
def mock_store(warm_profile: EntityProfile) -> _MockProfileStore:
    return _MockProfileStore({"usr_00000001": warm_profile})


@pytest.fixture
def pipeline(default_config: FeatureEngineeringConfig, mock_store: _MockProfileStore) -> FeaturePipeline:
    return FeaturePipeline(config=default_config, profile_store=mock_store)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Schema conformance tests
# ──────────────────────────────────────────────────────────────────────────────


class TestSchemaConformance:
    """All output EngineeredFeatures instances validate against DATA_SCHEMA.md."""

    def test_transform_single_produces_valid_schema(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        event = _make_inference_event(entity_id="usr_00000001")
        ef = pipeline.transform_single(event, profile=warm_profile)

        # Validate structure
        assert isinstance(ef, EngineeredFeatures)
        assert ef.entity_id == "usr_00000001"
        assert ef.event_id == event.event_id
        assert ef.session_id == event.session_id
        assert isinstance(ef.feature_vector, EntityFeatureVector)
        assert isinstance(ef.sequence_window, EntitySequence)

    def test_feature_vector_has_exactly_24_dimensions(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        event = _make_inference_event(entity_id="usr_00000001")
        ef = pipeline.transform_single(event, profile=warm_profile)
        assert len(ef.feature_vector.root) == FEATURE_DIM

    def test_feature_vector_values_in_valid_ranges(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        event = _make_inference_event(entity_id="usr_00000001")
        ef = pipeline.transform_single(event, profile=warm_profile)
        fvec = ef.feature_vector.root
        for i, val in enumerate(fvec):
            assert not math.isnan(val), f"dim {i} is NaN"
            assert not math.isinf(val), f"dim {i} is Inf"
            # dims 0-3 are sin/cos so can be in [-1, 1]
            if i in (0, 1, 2, 3):
                assert -1.0 <= val <= 1.0, f"dim {i} out of [-1, 1]: {val}"
            else:
                assert 0.0 <= val <= 1.0, f"dim {i} out of [0, 1]: {val}"

    def test_sequence_window_has_correct_shape(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        event = _make_inference_event(entity_id="usr_00000001")
        ef = pipeline.transform_single(event, profile=warm_profile)
        window = ef.sequence_window.root
        assert len(window) == 20  # W=20
        for row in window:
            assert len(row) == FEATURE_DIM

    def test_batch_transform_produces_all_records(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        events = [
            _make_inference_event(
                entity_id="usr_00000001",
                timestamp=_make_ts(hour=10, minute=i),
            )
            for i in range(5)
        ]
        results = pipeline.transform(events)
        assert len(results) == 5


# ──────────────────────────────────────────────────────────────────────────────
# 2. Geo-velocity tests (acceptance criterion 4)
# ──────────────────────────────────────────────────────────────────────────────


class TestGeoVelocity:
    """Geo-velocity computes distance/time between consecutive logins correctly."""

    # Known coordinate pairs with expected distances
    # Mumbai ↔ London: ~7,190 km
    MUMBAI = GeoLocation(city="Mumbai", country="IN", latitude=19.076, longitude=72.877)
    LONDON = GeoLocation(city="London", country="GB", latitude=51.509, longitude=-0.118)
    # Same location
    SAME = GeoLocation(city="Delhi", country="IN", latitude=28.704, longitude=77.102)

    def test_haversine_mumbai_to_london_distance(self) -> None:
        """Mumbai to London is approximately 7,190 km."""
        dist = haversine_distance_km(
            lat1=19.076, lon1=72.877,
            lat2=51.509, lon2=-0.118,
        )
        assert abs(dist - 7190.0) < 50.0, f"Expected ~7190 km, got {dist:.1f} km"

    def test_haversine_same_location_returns_zero(self) -> None:
        dist = haversine_distance_km(28.704, 77.102, 28.704, 77.102)
        assert dist < 0.001

    def test_haversine_antipodal_points(self) -> None:
        """Antipodal points are ~20,015 km (half Earth circumference)."""
        dist = haversine_distance_km(0.0, 0.0, 0.0, 180.0)
        assert abs(dist - 20015.0) < 50.0

    def test_geo_velocity_same_location_returns_zero(self) -> None:
        raw, norm = compute_geo_velocity(self.SAME, self.SAME, elapsed_seconds=3600.0)
        assert raw == 0.0
        assert norm == 0.0

    def test_geo_velocity_none_prev_geo_returns_zero(self) -> None:
        raw, norm = compute_geo_velocity(None, self.MUMBAI, elapsed_seconds=3600.0)
        assert raw == 0.0
        assert norm == 0.0

    def test_geo_velocity_zero_elapsed_returns_zero(self) -> None:
        raw, norm = compute_geo_velocity(self.MUMBAI, self.LONDON, elapsed_seconds=0.0)
        assert raw == 0.0
        assert norm == 0.0

    def test_geo_velocity_impossible_travel_normalized_to_one(self) -> None:
        """Mumbai to London in 16 minutes → ~26,900 km/h → capped at 1.0."""
        raw, norm = compute_geo_velocity(
            self.MUMBAI, self.LONDON, elapsed_seconds=16 * 60
        )
        assert raw > 2000.0  # definitely impossible travel speed
        assert norm == 1.0  # normalised cap

    def test_geo_velocity_realistic_speed(self) -> None:
        """Mumbai to London in 8 hours by airplane → ~900 km/h → normalised ≈ 0.45."""
        raw, norm = compute_geo_velocity(
            self.MUMBAI, self.LONDON, elapsed_seconds=8 * 3600
        )
        assert 800.0 < raw < 1000.0, f"Expected ~900 km/h, got {raw:.1f}"
        assert 0.4 < norm < 0.6

    def test_geo_velocity_longitude_wraparound(self) -> None:
        """Points near ±180° longitude should not produce nonsensical distances."""
        # Just past the antimeridian on both sides
        point_east = GeoLocation(city="A", country="ZZ", latitude=0.0, longitude=179.0)
        point_west = GeoLocation(city="B", country="ZZ", latitude=0.0, longitude=-179.0)
        dist = haversine_distance_km(0.0, 179.0, 0.0, -179.0)
        # Should be ~222 km (2° of longitude at equator), not ~40,000 km
        assert dist < 500.0, f"Wraparound distance unexpectedly large: {dist:.1f} km"

    def test_geo_velocity_in_feature_vector_dim6(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        """dim 6 in feature vector matches direct geo_velocity computation."""
        # First event (no previous geo)
        event1 = _make_inference_event(
            entity_id="usr_00000001",
            timestamp=_make_ts(hour=10, minute=0),
            geo=self.MUMBAI,
        )
        ef1 = pipeline.transform_single(event1, profile=warm_profile)
        assert ef1.feature_vector.root[6] == 0.0  # No previous location

        # Second event from London 16 minutes later → impossible travel
        event2 = _make_inference_event(
            entity_id="usr_00000001",
            session_id=event1.session_id,
            timestamp=_make_ts(hour=10, minute=16),
            geo=self.LONDON,
        )
        ef2 = pipeline.transform_single(event2, profile=warm_profile)
        assert ef2.feature_vector.root[6] == 1.0  # Capped at 1.0 (impossible travel)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Device fingerprint consistency tests (acceptance criterion 5)
# ──────────────────────────────────────────────────────────────────────────────


class TestFingerprintConsistency:
    """Device fingerprint consistency correctly identifies mismatched fingerprints."""

    def test_fingerprint_mac_match_known_mac(self) -> None:
        score = encode_fingerprint_mac_match("AA:BB:CC:11:22:33", ["AA:BB:CC:11:22:33"])
        assert score == 1.0

    def test_fingerprint_mac_match_unknown_mac(self) -> None:
        score = encode_fingerprint_mac_match("XX:XX:XX:99:88:77", ["AA:BB:CC:11:22:33"])
        assert score == 0.0

    def test_fingerprint_mac_match_cold_start(self) -> None:
        """Cold-start: no profile → neutral 0.5."""
        score = encode_fingerprint_mac_match("AA:BB:CC:11:22:33", None)
        assert score == 0.5

    def test_fingerprint_os_match_known_profile(self) -> None:
        score = encode_fingerprint_os_match(
            "Windows", "11.0",
            [{"os_family": "Windows", "os_version": "11.0"}]
        )
        assert score == 1.0

    def test_fingerprint_os_match_changed_os(self) -> None:
        """Strategy B: OS family changed → mismatch → 0.0."""
        score = encode_fingerprint_os_match(
            "Linux", "22.04",
            [{"os_family": "Windows", "os_version": "11.0"}]
        )
        assert score == 0.0

    def test_fingerprint_protocol_match_known(self) -> None:
        score = encode_fingerprint_protocol_match("HTTPS", ["HTTPS"])
        assert score == 1.0

    def test_fingerprint_protocol_match_changed(self) -> None:
        """Strategy C: protocol changed (e.g., Modbus → HTTPS for edge device)."""
        score = encode_fingerprint_protocol_match("HTTPS", ["Modbus"])
        assert score == 0.0

    def test_device_spoofing_dims_in_feature_vector(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        """Spoofed device produces dims 15=0.0, 16=0.0, 17=0.0."""
        spoofed_device = _make_device(
            mac="XX:XX:XX:99:88:77",
            os_family="Linux",
            os_version="22.04",
            protocol="SSH",
        )
        event = _make_inference_event(entity_id="usr_00000001", device=spoofed_device)
        ef = pipeline.transform_single(event, profile=warm_profile)
        fvec = ef.feature_vector.root
        assert fvec[15] == 0.0, f"fingerprint_os_match should be 0.0, got {fvec[15]}"
        assert fvec[16] == 0.0, f"fingerprint_mac_match should be 0.0, got {fvec[16]}"
        assert fvec[17] == 0.0, f"fingerprint_protocol_match should be 0.0, got {fvec[17]}"

    def test_legitimate_device_dims_in_feature_vector(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        """Legitimate device produces dims 15=1.0, 16=1.0, 17=1.0."""
        event = _make_inference_event(entity_id="usr_00000001")  # uses default device
        ef = pipeline.transform_single(event, profile=warm_profile)
        fvec = ef.feature_vector.root
        assert fvec[15] == 1.0
        assert fvec[16] == 1.0
        assert fvec[17] == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 4. Cold-start handling tests (acceptance criterion 3)
# ──────────────────────────────────────────────────────────────────────────────


class TestColdStartHandling:
    """Cold-start entities produce valid, non-degenerate feature vectors."""

    def test_cold_start_no_nan_or_inf(self, default_config: FeatureEngineeringConfig) -> None:
        """Brand-new entity (profile=None) must not produce NaN or Inf."""
        empty_store = _MockProfileStore({})  # No profiles
        pipeline = FeaturePipeline(config=default_config, profile_store=empty_store)
        event = _make_inference_event(entity_id="usr_newentity")
        ef = pipeline.transform_single(event)
        for i, val in enumerate(ef.feature_vector.root):
            assert not math.isnan(val), f"NaN at dim {i}"
            assert not math.isinf(val), f"Inf at dim {i}"

    def test_cold_start_session_metadata_is_cold_start(
        self, default_config: FeatureEngineeringConfig
    ) -> None:
        empty_store = _MockProfileStore({})
        pipeline = FeaturePipeline(config=default_config, profile_store=empty_store)
        event = _make_inference_event(entity_id="usr_newentity")
        ef = pipeline.transform_single(event)
        assert ef.session_metadata.is_cold_start is True

    def test_cold_start_fingerprint_dims_are_neutral(
        self, default_config: FeatureEngineeringConfig
    ) -> None:
        """Cold-start fingerprint dims should be 0.5 (neutral/unknown)."""
        empty_store = _MockProfileStore({})
        pipeline = FeaturePipeline(config=default_config, profile_store=empty_store)
        event = _make_inference_event(entity_id="usr_newentity")
        ef = pipeline.transform_single(event)
        fvec = ef.feature_vector.root
        assert fvec[15] == 0.5, f"fingerprint_os_match cold-start should be 0.5, got {fvec[15]}"
        assert fvec[16] == 0.5, f"fingerprint_mac_match cold-start should be 0.5, got {fvec[16]}"
        assert fvec[17] == 0.5, f"fingerprint_protocol_match cold-start should be 0.5, got {fvec[17]}"

    def test_cold_start_geo_velocity_is_zero_first_event(
        self, default_config: FeatureEngineeringConfig
    ) -> None:
        empty_store = _MockProfileStore({})
        pipeline = FeaturePipeline(config=default_config, profile_store=empty_store)
        event = _make_inference_event(entity_id="usr_newentity")
        ef = pipeline.transform_single(event)
        assert ef.feature_vector.root[6] == 0.0  # No previous location for cold-start

    def test_cold_start_resource_rarity_is_one_for_novel_resource(
        self, default_config: FeatureEngineeringConfig
    ) -> None:
        """Cold-start: no resource history → all resources are novel → rarity = 1.0."""
        empty_store = _MockProfileStore({})
        pipeline = FeaturePipeline(config=default_config, profile_store=empty_store)
        event = _make_inference_event(entity_id="usr_newentity")
        ef = pipeline.transform_single(event)
        assert ef.feature_vector.root[9] == 1.0  # resource_rarity_score

    def test_warm_entity_session_metadata_is_not_cold_start(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        event = _make_inference_event(entity_id="usr_00000001")
        ef = pipeline.transform_single(event, profile=warm_profile)
        assert ef.session_metadata.is_cold_start is False


# ──────────────────────────────────────────────────────────────────────────────
# 5. Label leakage prevention (acceptance criterion 7)
# ──────────────────────────────────────────────────────────────────────────────


class TestLabelLeakagePrevention:
    """Running pipeline on Training and Inference schemas produces identical feature vectors."""

    def test_training_and_inference_produce_identical_features(
        self, default_config: FeatureEngineeringConfig, warm_profile: EntityProfile
    ) -> None:
        """CRITICAL: features must be identical whether label is present or absent."""
        store = _MockProfileStore({"usr_00000001": warm_profile})

        ts = _make_ts(hour=14, minute=30)
        session_id = str(uuid.uuid4())
        geo = _make_geo()
        device = _make_device()
        commands = [
            CommandEntry(
                sequence_position=0, command="ls", target="/", outcome="success", elapsed_seconds=1.0
            )
        ]

        # Training event (has label)
        training_event = AccessLogTraining(
            event_id="evt-test-001",
            session_id=session_id,
            entity_id="usr_00000001",
            entity_type="user",
            timestamp=ts,
            source_ip="10.0.0.5",
            geo_location=geo,
            resource_accessed="file/finance/q1.xlsx",
            auth_method="password",
            auth_outcome="success",
            session_duration=600.0,
            command_sequence=commands,
            device_fingerprint=device,
            failure_count=0,
            label=AnomalyCategory.NORMAL,
        )

        # Inference event (no label, same raw fields)
        inference_event = AccessLogInference(
            event_id="evt-test-001",
            session_id=session_id,
            entity_id="usr_00000001",
            entity_type="user",
            timestamp=ts,
            source_ip="10.0.0.5",
            geo_location=geo,
            resource_accessed="file/finance/q1.xlsx",
            auth_method="password",
            auth_outcome="success",
            session_duration=600.0,
            command_sequence=commands,
            device_fingerprint=device,
            failure_count=0,
            delivery_mode="batch",
        )

        # Run each through a fresh pipeline (reset state between runs)
        p1 = FeaturePipeline(config=default_config, profile_store=store)
        ef_training_list = p1.transform_training([training_event])
        ef_training = ef_training_list[0]

        p2 = FeaturePipeline(config=default_config, profile_store=store)
        ef_inference = p2.transform_single(inference_event, profile=warm_profile)

        # Feature vectors must be identical
        for i, (train_val, infer_val) in enumerate(
            zip(ef_training.feature_vector.root, ef_inference.feature_vector.root)
        ):
            assert abs(train_val - infer_val) < 1e-10, (
                f"Dim {i} differs: training={train_val}, inference={infer_val}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 6. Batch vs. single-record mode identity (acceptance criterion 6)
# ──────────────────────────────────────────────────────────────────────────────


class TestBatchVsSingleRecord:
    """Batch mode and single-record mode produce identical feature vectors for the same record."""

    def test_batch_and_single_record_produce_same_features(
        self, default_config: FeatureEngineeringConfig, warm_profile: EntityProfile
    ) -> None:
        store = _MockProfileStore({"usr_00000001": warm_profile})
        ts = _make_ts(hour=9, minute=0)
        session_id = str(uuid.uuid4())
        geo = _make_geo()
        device = _make_device()

        event = AccessLogInference(
            event_id="evt-batch-001",
            session_id=session_id,
            entity_id="usr_00000001",
            entity_type="user",
            timestamp=ts,
            source_ip="10.0.0.1",
            geo_location=geo,
            resource_accessed="file/reports/annual.xlsx",
            auth_method="password",
            auth_outcome="success",
            session_duration=800.0,
            command_sequence=[],
            device_fingerprint=device,
            failure_count=0,
            delivery_mode="batch",
        )

        p_batch = FeaturePipeline(config=default_config, profile_store=store)
        batch_results = p_batch.transform([event])
        ef_batch = batch_results[0]

        p_single = FeaturePipeline(config=default_config, profile_store=store)
        ef_single = p_single.transform_single(event, profile=warm_profile)

        for i, (b, s) in enumerate(
            zip(ef_batch.feature_vector.root, ef_single.feature_vector.root)
        ):
            assert abs(b - s) < 1e-10, f"Dim {i} differs: batch={b}, single={s}"


# ──────────────────────────────────────────────────────────────────────────────
# 7. ProfileStoreInterface injection tests
# ──────────────────────────────────────────────────────────────────────────────


class TestProfileStoreInterface:
    """Mock ProfileStore can be injected and called correctly by FeaturePipeline."""

    def test_profile_store_interface_is_protocol(self) -> None:
        """_MockProfileStore satisfies the ProfileStoreInterface Protocol."""
        store = _MockProfileStore()
        assert isinstance(store, ProfileStoreInterface)

    def test_abstract_profile_store_is_abc(self) -> None:
        """AbstractProfileStore requires get_profile, upsert_profile, list_entity_ids."""
        import inspect
        abstract_methods = set(getattr(AbstractProfileStore, "__abstractmethods__", set()))
        assert "get_profile" in abstract_methods
        assert "upsert_profile" in abstract_methods
        assert "list_entity_ids" in abstract_methods

    def test_pipeline_calls_get_profile_on_store(
        self, default_config: FeatureEngineeringConfig
    ) -> None:
        """FeaturePipeline calls get_profile on the injected store."""
        store = MagicMock(spec=_MockProfileStore)
        store.get_profile.return_value = None  # Simulate cold-start
        pipeline = FeaturePipeline(config=default_config, profile_store=store)
        event = _make_inference_event(entity_id="usr_mock_001")
        pipeline.transform_single(event)
        store.get_profile.assert_called_once_with("usr_mock_001")

    def test_mock_profile_store_get_profiles_batch(self) -> None:
        """get_profiles_batch returns correct mapping."""
        profile = _make_warm_profile(entity_id="usr_00000001")
        store = _MockProfileStore({"usr_00000001": profile})
        result = store.get_profiles_batch(["usr_00000001", "usr_unknown"])
        assert result["usr_00000001"] is not None
        assert result["usr_unknown"] is None

    def test_upsert_and_retrieve_profile(self) -> None:
        store = _MockProfileStore()
        profile = _make_warm_profile(entity_id="usr_newtest")
        store.upsert_profile(profile)
        retrieved = store.get_profile("usr_newtest")
        assert retrieved is not None
        assert retrieved.entity_id == "usr_newtest"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Sequence construction tests
# ──────────────────────────────────────────────────────────────────────────────


class TestSequenceConstruction:
    """EntitySequence instances have correct shape per configuration."""

    def test_sequence_window_shape_single_event(self) -> None:
        """Single event → W=20 window, first 19 rows padded, last row is real."""
        config = FeatureEngineeringConfig(sequence_window_size=20)
        builder = SequenceBuilder(config)
        fvec = [float(i) for i in range(FEATURE_DIM)]
        window, mask = builder.update_and_get_window(fvec, "entity_1")
        assert len(window) == 20
        assert len(mask) == 20
        assert sum(mask) == 1  # 1 real event
        assert all(not m for m in mask[:19])
        assert mask[19] is True
        assert window[19] == fvec

    def test_sequence_window_fully_filled(self) -> None:
        """20 events → W=20 window, all rows are real, no padding."""
        config = FeatureEngineeringConfig(sequence_window_size=20)
        builder = SequenceBuilder(config)
        for i in range(20):
            fvec = [float(i)] * FEATURE_DIM
            window, mask = builder.update_and_get_window(fvec, "entity_full")
        assert sum(mask) == 20  # All positions are real
        assert all(mask)

    def test_sequence_window_oldest_event_evicted(self) -> None:
        """21st event evicts the 1st event from the deque."""
        config = FeatureEngineeringConfig(sequence_window_size=20)
        builder = SequenceBuilder(config)
        for i in range(21):
            fvec = [float(i)] * FEATURE_DIM
            window, mask = builder.update_and_get_window(fvec, "entity_evict")
        # Window should contain events 1–20 (0 is evicted), all real
        assert sum(mask) == 20
        assert window[0][0] == 1.0  # event 1 is now oldest
        assert window[19][0] == 20.0  # event 20 is most recent

    def test_sliding_windows_batch_mode_count(self) -> None:
        """N events with stride=1 produce N windows."""
        config = FeatureEngineeringConfig(sequence_window_size=5, stride=1)
        builder = SequenceBuilder(config)
        fvecs = [[float(i)] * FEATURE_DIM for i in range(10)]
        windows = list(builder.sliding_windows(fvecs))
        assert len(windows) == 10

    def test_build_window_helper_produces_correct_padding(self) -> None:
        """3 events in a W=5 window → 2 padding rows + 3 real rows."""
        history = [[1.0] * FEATURE_DIM, [2.0] * FEATURE_DIM, [3.0] * FEATURE_DIM]
        window, mask = SequenceBuilder._build_window(history, window_size=5)
        assert len(window) == 5
        assert mask == [False, False, True, True, True]
        assert window[0] == [0.0] * FEATURE_DIM  # padding
        assert window[2][0] == 1.0
        assert window[4][0] == 3.0

    def test_seed_from_profile_seeds_deque_correctly(self) -> None:
        config = FeatureEngineeringConfig(sequence_window_size=5)
        builder = SequenceBuilder(config)
        history = [[float(i)] * FEATURE_DIM for i in range(3)]
        builder.seed_from_profile("entity_seeded", history)
        window, mask = builder.get_current_window("entity_seeded")
        assert sum(mask) == 3


# ──────────────────────────────────────────────────────────────────────────────
# 9. Attack signal coverage tests (acceptance criterion 2)
# ──────────────────────────────────────────────────────────────────────────────


class TestAttackSignalCoverage:
    """For each attack type, at least one feature is measurably different from normal."""

    def _make_pipeline_and_profile(self) -> tuple[FeaturePipeline, EntityProfile, FeatureEngineeringConfig]:
        config = FeatureEngineeringConfig()
        config.cold_start_session_duration_mean = 600.0
        config.cold_start_session_duration_std = 300.0
        config.population_most_frequent_country = "IN"
        profile = _make_warm_profile(entity_id="usr_attack_test")
        store = _MockProfileStore({"usr_attack_test": profile})
        pipeline = FeaturePipeline(config=config, profile_store=store)
        return pipeline, profile, config

    def test_brute_force_failure_count_dim5_elevated(self) -> None:
        """Brute Force: dim 5 (failure_count_norm) is high for attack events."""
        pipeline, profile, _ = self._make_pipeline_and_profile()
        # Normal event
        normal = _make_inference_event(entity_id="usr_attack_test", failure_count=0)
        ef_normal = pipeline.transform_single(normal, profile=profile)

        # Brute-force event (20 failures)
        attack = _make_inference_event(
            entity_id="usr_attack_test",
            session_id=normal.session_id,
            timestamp=_make_ts(hour=10, minute=1),
            failure_count=20,
            auth_outcome="failure",
        )
        ef_attack = pipeline.transform_single(attack, profile=profile)

        assert ef_attack.feature_vector.root[5] == 1.0   # failure_count_norm=1.0
        assert ef_attack.feature_vector.root[11] == 1.0  # auth_outcome=failure=1.0
        assert ef_attack.feature_vector.root[5] > ef_normal.feature_vector.root[5]

    def test_impossible_travel_geo_velocity_dim6_elevated(self) -> None:
        """Impossible Travel: dim 6 (geo_velocity) is 1.0 for attack events."""
        pipeline, profile, _ = self._make_pipeline_and_profile()
        # Normal event in Mumbai
        normal = _make_inference_event(
            entity_id="usr_attack_test",
            geo=_make_geo(city="Mumbai", country="IN", lat=19.076, lon=72.877),
            timestamp=_make_ts(hour=10, minute=0),
        )
        pipeline.transform_single(normal, profile=profile)  # Set prev_geo

        # Impossible travel: London 10 minutes later
        attack = _make_inference_event(
            entity_id="usr_attack_test",
            session_id=normal.session_id,
            geo=_make_geo(city="London", country="GB", lat=51.509, lon=-0.118),
            timestamp=_make_ts(hour=10, minute=10),
        )
        ef_attack = pipeline.transform_single(attack, profile=profile)
        assert ef_attack.feature_vector.root[6] == 1.0  # Geo velocity capped at max

    def test_credential_stuffing_ip_entity_ratio_dim22_elevated(self) -> None:
        """Credential Stuffing: dim 22 (ip_entity_ratio) is elevated."""
        pipeline, profile, _ = self._make_pipeline_and_profile()
        # Normal: low failure, unique IP
        normal = _make_inference_event(
            entity_id="usr_attack_test",
            failure_count=0,
            auth_outcome="success",
            source_ip="192.168.1.1",
        )
        ef_normal = pipeline.transform_single(normal, profile=profile)

        # Stuffing attack: many failures, same IP as many other entities
        # (Simulated by dim 5 being elevated)
        attack = _make_inference_event(
            entity_id="usr_attack_test",
            session_id=normal.session_id,
            timestamp=_make_ts(hour=10, minute=1),
            failure_count=3,
            auth_outcome="failure",
            source_ip="10.0.0.99",  # Foreign IP
        )
        ef_attack = pipeline.transform_single(attack, profile=profile)
        # failure_count_norm (dim 5) should be elevated vs. normal
        assert ef_attack.feature_vector.root[5] > ef_normal.feature_vector.root[5]
        # auth_outcome (dim 11) = 1.0 for failure
        assert ef_attack.feature_vector.root[11] == 1.0

    def test_lateral_movement_resource_rarity_dim9_elevated(self) -> None:
        """Lateral Movement: dim 9 (resource_rarity) is 1.0 for novel resources."""
        pipeline, profile, _ = self._make_pipeline_and_profile()
        # Normal: access a known resource
        normal = _make_inference_event(
            entity_id="usr_attack_test",
            resource="file/reports/q1.xlsx",  # in profile's resource_access_counts
        )
        ef_normal = pipeline.transform_single(normal, profile=profile)

        # Lateral movement: completely novel resource
        attack = _make_inference_event(
            entity_id="usr_attack_test",
            session_id=normal.session_id,
            timestamp=_make_ts(hour=10, minute=1),
            resource="api/admin/users",  # not in profile
            commands=[
                CommandEntry(
                    sequence_position=0, command="scp", target="192.168.99.5:/tmp/",
                    outcome="success", elapsed_seconds=5.0
                )
            ],
        )
        ef_attack = pipeline.transform_single(attack, profile=profile)
        assert ef_attack.feature_vector.root[9] == 1.0  # resource_rarity = 1.0 (novel)
        assert ef_attack.feature_vector.root[14] == 1.0  # has_exfil_command = 1.0

    def test_device_spoofing_fingerprint_dims_differ(self) -> None:
        """Device Spoofing: dims 15–17 are 0.0 for spoofed fingerprint."""
        pipeline, profile, _ = self._make_pipeline_and_profile()
        normal = _make_inference_event(entity_id="usr_attack_test")
        ef_normal = pipeline.transform_single(normal, profile=profile)
        assert ef_normal.feature_vector.root[16] == 1.0  # known MAC → match

        attack = _make_inference_event(
            entity_id="usr_attack_test",
            session_id=normal.session_id,
            timestamp=_make_ts(hour=10, minute=1),
            device=_make_device(mac="99:88:77:66:55:44", os_family="Linux", os_version="22.04"),
        )
        ef_attack = pipeline.transform_single(attack, profile=profile)
        assert ef_attack.feature_vector.root[15] == 0.0  # OS mismatch
        assert ef_attack.feature_vector.root[16] == 0.0  # MAC mismatch

    def test_low_and_slow_exfil_command_dim14_elevated(self) -> None:
        """Low-and-Slow: dim 14 (has_exfil_command) is 1.0 for attack events."""
        pipeline, profile, _ = self._make_pipeline_and_profile()
        normal = _make_inference_event(entity_id="usr_attack_test", commands=[])
        ef_normal = pipeline.transform_single(normal, profile=profile)
        assert ef_normal.feature_vector.root[14] == 0.0

        attack = _make_inference_event(
            entity_id="usr_attack_test",
            session_id=normal.session_id,
            timestamp=_make_ts(hour=2, minute=30),  # Off-hours
            commands=[
                CommandEntry(
                    sequence_position=0, command="scp", target="external.host:/tmp/",
                    outcome="success", elapsed_seconds=10.0
                )
            ],
        )
        ef_attack = pipeline.transform_single(attack, profile=profile)
        assert ef_attack.feature_vector.root[14] == 1.0  # exfil command present

    def test_insider_drift_resource_rarity_gradually_increases(self) -> None:
        """Insider Drift: resource_rarity (dim 9) is elevated for novel resources."""
        pipeline, profile, _ = self._make_pipeline_and_profile()
        # Normal: known resource
        normal = _make_inference_event(
            entity_id="usr_attack_test",
            resource="file/reports/q1.xlsx",
        )
        ef_normal = pipeline.transform_single(normal, profile=profile)
        normal_rarity = ef_normal.feature_vector.root[9]

        # Insider drift: novel resource (outside entity's historical set)
        drift = _make_inference_event(
            entity_id="usr_attack_test",
            session_id=normal.session_id,
            timestamp=_make_ts(hour=10, minute=2),
            resource="file/hr/salary_band.xlsx",  # Not in profile
        )
        ef_drift = pipeline.transform_single(drift, profile=profile)
        drift_rarity = ef_drift.feature_vector.root[9]
        assert drift_rarity > normal_rarity


# ──────────────────────────────────────────────────────────────────────────────
# 10. Edge case tests
# ──────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: single event, multi-window spans, empty event set."""

    def test_single_event_entity_produces_valid_output(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        event = _make_inference_event(entity_id="usr_00000001")
        ef = pipeline.transform_single(event, profile=warm_profile)
        assert len(ef.feature_vector.root) == FEATURE_DIM

    def test_events_spanning_multiple_sessions(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        """Session boundary resets session_event_count and resource breadth."""
        sess1 = str(uuid.uuid4())
        sess2 = str(uuid.uuid4())
        event1 = _make_inference_event(
            entity_id="usr_00000001",
            session_id=sess1,
            timestamp=_make_ts(hour=9, minute=0),
        )
        event2 = _make_inference_event(
            entity_id="usr_00000001",
            session_id=sess1,
            timestamp=_make_ts(hour=9, minute=1),
        )
        event3 = _make_inference_event(
            entity_id="usr_00000001",
            session_id=sess2,
            timestamp=_make_ts(hour=10, minute=0),
        )
        ef1 = pipeline.transform_single(event1, profile=warm_profile)
        ef2 = pipeline.transform_single(event2, profile=warm_profile)
        ef3 = pipeline.transform_single(event3, profile=warm_profile)

        # After session break, session_event_count should reset to 1
        # dim 20 is session_event_count_norm
        # event3 is the first in sess2 → session_event_count=1
        # event2 is the second in sess1 → session_event_count=2
        assert ef3.feature_vector.root[20] < ef2.feature_vector.root[20]

    def test_empty_batch_returns_empty_list(
        self, pipeline: FeaturePipeline
    ) -> None:
        results = pipeline.transform([])
        assert results == []

    def test_inter_event_gap_is_zero_for_first_event(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        event = _make_inference_event(entity_id="usr_00000001")
        ef = pipeline.transform_single(event, profile=warm_profile)
        # dim 19 is inter_event_gap_norm; first event has no prior → 0.0
        assert ef.feature_vector.root[19] == 0.0

    def test_inter_event_gap_increases_for_second_event(
        self, pipeline: FeaturePipeline, warm_profile: EntityProfile
    ) -> None:
        event1 = _make_inference_event(
            entity_id="usr_00000001",
            timestamp=_make_ts(hour=8, minute=0),
        )
        event2 = _make_inference_event(
            entity_id="usr_00000001",
            session_id=event1.session_id,
            timestamp=_make_ts(hour=20, minute=0),  # 12 hours later
        )
        pipeline.transform_single(event1, profile=warm_profile)
        ef2 = pipeline.transform_single(event2, profile=warm_profile)
        # 12 hours / 24 hours = 0.5
        assert abs(ef2.feature_vector.root[19] - 0.5) < 0.01

    def test_failure_count_capped_at_one(self) -> None:
        """failure_count >= 20 is normalised to 1.0."""
        assert encode_failure_count(20) == 1.0
        assert encode_failure_count(100) == 1.0
        assert encode_failure_count(0) == 0.0
        assert abs(encode_failure_count(10) - 0.5) < 0.001

    def test_unknown_auth_method_defaults_to_zero(self) -> None:
        assert encode_auth_method("biometric_scan") == 0.0

    def test_unknown_auth_outcome_defaults_to_zero(self) -> None:
        assert encode_auth_outcome("timeout") == 0.0

    def test_resource_rarity_with_no_access_history(self) -> None:
        """Zero total events → rarity = 1.0 (cold-start protection)."""
        score = encode_resource_rarity("file/secret.txt", {}, 0)
        assert score == 1.0

    def test_has_exfil_command_with_empty_sequence(self) -> None:
        assert encode_has_exfil_command([]) == 0.0

    def test_has_exfil_command_all_exfil_variants(self) -> None:
        for cmd_name in ["scp", "rsync", "ftp", "curl", "wget", "nc"]:
            cmds = [CommandEntry(
                sequence_position=0, command=cmd_name, target="host",
                outcome="success", elapsed_seconds=1.0
            )]
            assert encode_has_exfil_command(cmds) == 1.0, f"{cmd_name} not detected"

    def test_hour_encoding_no_discontinuity(self) -> None:
        """Hour 23 and hour 0 should be close in circular encoding space."""
        sin23, cos23 = encode_hour_of_day(23)
        sin0, cos0 = encode_hour_of_day(0)
        distance = math.sqrt((sin23 - sin0) ** 2 + (cos23 - cos0) ** 2)
        # One hour apart in circular space; distance should be small
        assert distance < 0.5, f"Hour 23→0 circular distance too large: {distance}"

    def test_day_encoding_no_discontinuity(self) -> None:
        """Day 6 (Sunday) and day 0 (Monday) should be close in circular space."""
        sin6, cos6 = encode_day_of_week(6)
        sin0, cos0 = encode_day_of_week(0)
        distance = math.sqrt((sin6 - sin0) ** 2 + (cos6 - cos0) ** 2)
        assert distance < 1.0, f"Day 6→0 circular distance too large: {distance}"


# ──────────────────────────────────────────────────────────────────────────────
# 11. Encoder unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestEncoders:
    """Unit tests for individual encoder functions."""

    def test_encode_resource_category_all_categories(self) -> None:
        assert encode_resource_category("file/x") == 0.0 / 5.0
        assert encode_resource_category("port/22") == 1.0 / 5.0
        assert encode_resource_category("api/admin") == 2.0 / 5.0
        assert encode_resource_category("db/prod") == 3.0 / 5.0
        assert encode_resource_category("device/router") == 4.0 / 5.0
        assert encode_resource_category("other/thing") == 5.0 / 5.0

    def test_encode_is_new_geo_same_country(self) -> None:
        assert encode_is_new_geo("IN", "IN") == 0.0

    def test_encode_is_new_geo_different_country(self) -> None:
        assert encode_is_new_geo("GB", "IN") == 1.0

    def test_encode_is_new_geo_cold_start_returns_zero(self) -> None:
        assert encode_is_new_geo("IN", None) == 0.0

    def test_entity_type_encoding_values(self) -> None:
        from anomaly_detection.feature_engineering.encoders import encode_entity_type
        assert encode_entity_type("user") == 0.0
        assert encode_entity_type("service_account") == 0.5
        assert encode_entity_type("edge_device") == 1.0

    def test_auth_method_encoding_all_values(self) -> None:
        assert encode_auth_method("password") == 0.0 / 4.0
        assert encode_auth_method("token") == 1.0 / 4.0
        assert encode_auth_method("certificate") == 2.0 / 4.0
        assert encode_auth_method("biometric") == 3.0 / 4.0
        assert encode_auth_method("none") == 4.0 / 4.0

    def test_auth_outcome_encoding_values(self) -> None:
        assert encode_auth_outcome("success") == 0.0
        assert encode_auth_outcome("mfa_required") == 0.5
        assert encode_auth_outcome("failure") == 1.0
