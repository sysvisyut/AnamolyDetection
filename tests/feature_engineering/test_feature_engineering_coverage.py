"""
Additional coverage tests for FeaturePipeline.fit(), SequenceBuilder,
and uncovered branches in encoders and feature_extractor.

Tier: T1
Pytest mark: @pytest.mark.tier1
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

import pytest

from anomaly_detection.common.models.access_log import (
    AccessLogInference,
    AccessLogTraining,
    CommandEntry,
    DeviceFingerprint,
    GeoLocation,
)
from anomaly_detection.common.models.entities import EntityProfile
from anomaly_detection.common.models.enums import AnomalyCategory, EntityType
from anomaly_detection.feature_engineering import (
    FEATURE_DIM,
    FeatureEngineeringConfig,
    FeatureExtractor,
    FeaturePipeline,
    SequenceBuilder,
    SessionBuilder,
)
from anomaly_detection.feature_engineering.encoders import (
    encode_command_rarity,
    encode_command_seq_length,
    encode_entity_ip_ratio,
    encode_inter_event_gap,
    encode_ip_entity_ratio,
    encode_resource_breadth,
    encode_session_event_count,
)
from anomaly_detection.feature_engineering.sequence_builder import SequenceBuilder

pytestmark = pytest.mark.tier1


def _ts(hour: int = 10, minute: int = 0, day: int = 1) -> str:
    return f"2026-07-{day:02d}T{hour:02d}:{minute:02d}:00.000000+00:00"


def _geo(country: str = "IN", lat: float = 19.076, lon: float = 72.877) -> GeoLocation:
    return GeoLocation(city="Mumbai", country=country, latitude=lat, longitude=lon)


def _device(mac: str = "AA:BB:CC:11:22:33") -> DeviceFingerprint:
    return DeviceFingerprint(
        device_id="dev_001", os_family="Windows", os_version="11.0",
        mac_address=mac, protocol="HTTPS", user_agent="Mozilla/5.0", firmware_version="",
    )


def _inference(
    entity_id: str = "usr_001",
    entity_type: str = "user",
    timestamp: str = "",
    session_id: str = "",
    failure_count: int = 0,
    auth_outcome: str = "success",
    resource: str = "file/reports/q1.xlsx",
    label: str = "normal",
) -> AccessLogTraining:
    return AccessLogTraining(
        event_id=str(uuid.uuid4()),
        session_id=session_id or str(uuid.uuid4()),
        entity_id=entity_id,
        entity_type=entity_type,
        timestamp=timestamp or _ts(),
        source_ip="10.0.0.1",
        geo_location=_geo(),
        resource_accessed=resource,
        auth_method="password",
        auth_outcome=auth_outcome,
        session_duration=600.0,
        command_sequence=[],
        device_fingerprint=_device(),
        failure_count=failure_count,
        label=AnomalyCategory(label),
    )


class _MockStore:
    def __init__(self, profiles: Optional[Dict[str, EntityProfile]] = None) -> None:
        self._profiles: Dict[str, Optional[EntityProfile]] = dict(profiles or {})

    def get_profile(self, eid: str) -> Optional[EntityProfile]:
        return self._profiles.get(eid)

    def get_profiles_batch(self, eids: List[str]) -> Dict[str, Optional[EntityProfile]]:
        return {e: self.get_profile(e) for e in eids}

    def upsert_profile(self, profile: EntityProfile) -> None:
        self._profiles[profile.entity_id] = profile

    def list_entity_ids(self) -> List[str]:
        return sorted(self._profiles.keys())


class TestFeaturePipelineFit:
    """Tests for FeaturePipeline.fit() population statistics computation."""

    def test_fit_computes_population_baseline_vectors(self) -> None:
        """After fit(), population_baseline_vectors has entries per entity type."""
        cfg = FeatureEngineeringConfig()
        store = _MockStore()
        pipeline = FeaturePipeline(config=cfg, profile_store=store)

        records = [
            _inference(entity_id=f"usr_{i:04d}", timestamp=_ts(hour=9, minute=i))
            for i in range(5)
        ] + [
            _inference(entity_id=f"svc_{i:04d}", entity_type="service_account", timestamp=_ts(hour=10, minute=i))
            for i in range(5)
        ]
        pipeline.fit(records)
        assert "user" in cfg.population_baseline_vectors
        assert "service_account" in cfg.population_baseline_vectors
        assert len(cfg.population_baseline_vectors["user"]) == FEATURE_DIM
        assert len(cfg.population_baseline_vectors["service_account"]) == FEATURE_DIM

    def test_fit_computes_population_baseline_stds(self) -> None:
        cfg = FeatureEngineeringConfig()
        store = _MockStore()
        pipeline = FeaturePipeline(config=cfg, profile_store=store)
        records = [
            _inference(entity_id=f"usr_{i:04d}", timestamp=_ts(hour=9, minute=i))
            for i in range(10)
        ]
        pipeline.fit(records)
        assert "user" in cfg.population_baseline_stds
        stds = cfg.population_baseline_stds["user"]
        assert len(stds) == FEATURE_DIM
        # All stds must be >= 0
        assert all(s >= 0.0 for s in stds)

    def test_fit_computes_session_duration_mean_std(self) -> None:
        cfg = FeatureEngineeringConfig()
        store = _MockStore()
        pipeline = FeaturePipeline(config=cfg, profile_store=store)
        records = [
            _inference(entity_id=f"usr_{i:04d}", timestamp=_ts(hour=9, minute=i))
            for i in range(10)
        ]
        pipeline.fit(records)
        assert cfg.cold_start_session_duration_mean == 600.0
        assert cfg.cold_start_session_duration_std >= 1.0

    def test_fit_computes_population_most_frequent_country(self) -> None:
        cfg = FeatureEngineeringConfig()
        store = _MockStore()
        pipeline = FeaturePipeline(config=cfg, profile_store=store)
        records = [
            _inference(entity_id=f"usr_{i:04d}", timestamp=_ts(hour=9, minute=i))
            for i in range(5)
        ]
        pipeline.fit(records)
        # All events are from "IN" (default geo)
        assert cfg.population_most_frequent_country == "IN"

    def test_fit_does_not_include_attack_events_in_population(self) -> None:
        """Only normal-labeled events contribute to population baselines."""
        cfg = FeatureEngineeringConfig()
        store = _MockStore()
        pipeline = FeaturePipeline(config=cfg, profile_store=store)

        normal_records = [
            _inference(entity_id=f"usr_{i:04d}", timestamp=_ts(hour=9, minute=i), label="normal")
            for i in range(5)
        ]
        attack_records = [
            _inference(entity_id=f"usr_{i:04d}", timestamp=_ts(hour=10, minute=i), label="brute_force", failure_count=20, auth_outcome="failure")
            for i in range(5)
        ]
        pipeline.fit(normal_records + attack_records)
        # Population vectors should be in the normal range (dim 5 = failure_count should be 0)
        dim5_mean = cfg.population_baseline_vectors["user"][5]
        assert dim5_mean == 0.0, f"Attack events contaminated population: dim5_mean={dim5_mean}"

    def test_fit_resets_state_for_subsequent_transform(self) -> None:
        """After fit(), state is reset so transform() starts fresh."""
        cfg = FeatureEngineeringConfig()
        store = _MockStore()
        pipeline = FeaturePipeline(config=cfg, profile_store=store)

        records = [
            _inference(entity_id="usr_0001", timestamp=_ts(hour=9, minute=i))
            for i in range(3)
        ]
        pipeline.fit(records)
        # After fit, pipeline state is reset — transform should work from scratch
        inf_events = [
            AccessLogInference(
                event_id=r.event_id,
                session_id=r.session_id,
                entity_id=r.entity_id,
                entity_type=r.entity_type,
                timestamp=r.timestamp,
                source_ip=r.source_ip,
                geo_location=r.geo_location,
                resource_accessed=r.resource_accessed,
                auth_method=r.auth_method,
                auth_outcome=r.auth_outcome,
                session_duration=r.session_duration,
                command_sequence=r.command_sequence,
                device_fingerprint=r.device_fingerprint,
                failure_count=r.failure_count,
                delivery_mode="batch",
            )
            for r in records
        ]
        results = pipeline.transform(inf_events)
        assert len(results) == 3
        assert results[0].feature_vector.root[19] == 0.0  # First event: no gap


class TestSequenceBuilderExtended:
    """Extended coverage for SequenceBuilder methods."""

    def test_window_to_flat_produces_correct_length(self) -> None:
        window = [[float(i)] * FEATURE_DIM for i in range(20)]
        flat = SequenceBuilder.window_to_flat(window)
        assert len(flat) == 20 * FEATURE_DIM

    def test_window_to_flat_values_are_correct(self) -> None:
        window = [[1.0] * FEATURE_DIM, [2.0] * FEATURE_DIM]
        flat = SequenceBuilder.window_to_flat(window)
        assert flat[:FEATURE_DIM] == [1.0] * FEATURE_DIM
        assert flat[FEATURE_DIM:] == [2.0] * FEATURE_DIM

    def test_reset_clears_entity_windows(self) -> None:
        config = FeatureEngineeringConfig(sequence_window_size=5)
        builder = SequenceBuilder(config)
        fvec = [0.5] * FEATURE_DIM
        builder.update_and_get_window(fvec, "entity_a")
        builder.reset()
        window, mask = builder.get_current_window("entity_a")
        assert not any(mask)  # All padding after reset

    def test_build_batch_windows_returns_parallel_lists(self) -> None:
        config = FeatureEngineeringConfig(sequence_window_size=5, stride=1)
        builder = SequenceBuilder(config)
        fvecs = [[float(i)] * FEATURE_DIM for i in range(8)]
        windows, masks = builder.build_batch_windows(fvecs)
        assert len(windows) == 8
        assert len(masks) == 8
        assert all(len(w) == 5 for w in windows)

    def test_build_window_with_empty_history_returns_full_padding(self) -> None:
        window, mask = SequenceBuilder._build_window([], window_size=5)
        assert len(window) == 5
        assert not any(mask)
        # All zeros
        for row in window:
            assert all(v == 0.0 for v in row)

    def test_update_and_get_window_invalid_length_raises(self) -> None:
        config = FeatureEngineeringConfig(sequence_window_size=5)
        builder = SequenceBuilder(config)
        with pytest.raises(ValueError):
            builder.update_and_get_window([1.0, 2.0, 3.0], "entity_bad")


class TestEncodersCoverage:
    """Cover branches missed by primary test suite."""

    def test_encode_command_seq_length_empty(self) -> None:
        assert encode_command_seq_length([]) == 0.0

    def test_encode_command_seq_length_capped(self) -> None:
        commands = [
            CommandEntry(sequence_position=i, command="ls", target="/", outcome="success", elapsed_seconds=float(i))
            for i in range(100)
        ]
        assert encode_command_seq_length(commands) == 1.0

    def test_encode_command_rarity_empty_sequence_returns_neutral(self) -> None:
        score = encode_command_rarity([], {"ls": 5}, 10)
        assert score == 0.5  # cold_start_default

    def test_encode_command_rarity_no_history_returns_neutral(self) -> None:
        cmds = [CommandEntry(sequence_position=0, command="sudo", target="/", outcome="success", elapsed_seconds=1.0)]
        score = encode_command_rarity(cmds, None, 0)
        assert score == 0.5

    def test_encode_command_rarity_known_command_is_low(self) -> None:
        """A very frequent command should have low rarity."""
        cmds = [CommandEntry(sequence_position=0, command="ls", target="/", outcome="success", elapsed_seconds=1.0)]
        score = encode_command_rarity(cmds, {"ls": 100}, 100)
        # freq = 100/100 = 1.0, rarity = 0.0
        assert score == 0.0

    def test_encode_session_event_count_capped(self) -> None:
        assert encode_session_event_count(200) == 1.0
        assert encode_session_event_count(0) == 0.0

    def test_encode_resource_breadth_capped(self) -> None:
        assert encode_resource_breadth(50) == 1.0
        assert encode_resource_breadth(0) == 0.0

    def test_encode_ip_entity_ratio_zero_denominator(self) -> None:
        assert encode_ip_entity_ratio(5, 0) == 0.0

    def test_encode_entity_ip_ratio_zero_denominator(self) -> None:
        assert encode_entity_ip_ratio(3, 0) == 0.0

    def test_encode_inter_event_gap_capped(self) -> None:
        assert encode_inter_event_gap(86400.0) == 1.0
        assert encode_inter_event_gap(0.0) == 0.0
        assert abs(encode_inter_event_gap(43200.0) - 0.5) < 0.001  # 12 hours

    def test_encode_ip_entity_ratio_high_ratio_capped(self) -> None:
        # 100 events from this IP for an entity with 10 events → ratio = 10.0 → capped
        score = encode_ip_entity_ratio(100, 10)
        assert score == 1.0


class TestSessionBuilderReset:
    """Verify SessionBuilder.reset() clears all state."""

    def test_reset_clears_entity_states(self) -> None:
        config = FeatureEngineeringConfig()
        builder = SessionBuilder(config)
        # Process an event to populate state
        from tests.feature_engineering.test_feature_engineering import _make_inference_event
        event = _make_inference_event(entity_id="usr_test_reset")
        builder.process_event(event)
        assert "usr_test_reset" in builder.get_entity_ids()
        builder.reset()
        assert "usr_test_reset" not in builder.get_entity_ids()

    def test_sort_events_chronologically(self) -> None:
        from tests.feature_engineering.test_feature_engineering import _make_inference_event
        events = [
            _make_inference_event(entity_id="usr_001", timestamp=_ts(hour=12)),
            _make_inference_event(entity_id="usr_001", timestamp=_ts(hour=9)),
            _make_inference_event(entity_id="usr_001", timestamp=_ts(hour=15)),
        ]
        sorted_events = SessionBuilder.sort_events_chronologically(events)
        ts_list = [e.timestamp for e in sorted_events]
        assert ts_list == sorted(ts_list)


class TestFeatureExtractorDirectly:
    """Direct FeatureExtractor tests for uncovered branches."""

    def test_extract_warm_profile_with_session_duration_normalisation(self) -> None:
        """session_duration_norm uses entity baseline for warm entities."""
        config = FeatureEngineeringConfig()
        config.cold_start_session_duration_mean = 600.0
        config.cold_start_session_duration_std = 100.0
        extractor = FeatureExtractor(config)

        # A warm profile with baseline_vector[4] = 600.0 (meaning entity's mean is 600s)
        profile = EntityProfile(
            entity_id="usr_warm",
            entity_type=EntityType.USER,
            baseline_vector=[0.5] * 4 + [600.0] + [0.5] * 19,
            baseline_std=[0.1] * 4 + [100.0] + [0.1] * 19,
            sequence_history=[],
            most_frequent_country="IN",
            known_mac_addresses=["AA:BB:CC:11:22:33"],
            known_os_profiles=[{"os_family": "Windows", "os_version": "11.0"}],
            known_protocols=["HTTPS"],
            resource_access_counts={},
            command_frequency={},
            event_count=50,
            cold_start_flag=False,
            last_updated=_ts(),
            profile_version=1,
        )

        from tests.feature_engineering.test_feature_engineering import (
            _make_inference_event,
            _MockProfileStore,
        )
        store = _MockProfileStore({"usr_warm": profile})
        pipeline = FeaturePipeline(config=config, profile_store=store)
        event = _make_inference_event(entity_id="usr_warm")
        ef = pipeline.transform_single(event, profile=profile)
        # At exactly the mean (600.0), deviation = 0 → session_duration_norm = 0.0
        assert ef.feature_vector.root[4] == 0.0

    def test_extract_disabled_feature_flags_produce_zeros(self) -> None:
        """When feature flags are disabled, corresponding dims are 0.0."""
        config = FeatureEngineeringConfig(
            enable_geo_features=False,
            enable_device_features=False,
            enable_command_features=False,
            enable_ratio_features=False,
        )
        store = _MockStore()
        pipeline = FeaturePipeline(config=config, profile_store=store)
        from tests.feature_engineering.test_feature_engineering import _make_inference_event
        event = _make_inference_event(entity_id="usr_flags_test")
        ef = pipeline.transform_single(event)
        fvec = ef.feature_vector.root
        assert fvec[6] == 0.0  # geo disabled
        assert fvec[7] == 0.0
        assert fvec[12] == 0.0  # command disabled
        assert fvec[13] == 0.0
        assert fvec[14] == 0.0
        assert fvec[22] == 0.0  # ratio disabled
        assert fvec[23] == 0.0
