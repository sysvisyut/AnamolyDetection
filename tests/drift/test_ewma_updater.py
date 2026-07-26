"""Tests for ``drift.ewma_updater``.

Tier: T2
Pytest mark: @pytest.mark.tier2
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from anomaly_detection.common.models.entities import EntityProfile
from anomaly_detection.common.models.enums import EntityType

from src.drift import DriftConfig, EWMAUpdater


@dataclass
class InMemoryProfileStore:
    """Small injected store used to observe updater writes."""

    profile: EntityProfile | None
    upsert_count: int = 0

    def get_profile(self, entity_id: str) -> EntityProfile | None:
        """Return the stored profile only for its matching entity."""
        if self.profile is not None and self.profile.entity_id == entity_id:
            return self.profile
        return None

    def upsert_profile(self, profile: EntityProfile) -> None:
        """Record and retain an updater write."""
        self.profile = profile
        self.upsert_count += 1


@pytest.fixture
def sample_profile() -> EntityProfile:
    """Provide a warm profile with a three-value baseline for exact arithmetic."""
    return EntityProfile(
        entity_id="usr_001",
        entity_type=EntityType.USER,
        baseline_vector=[1.0, 2.0, 3.0],
        baseline_std=[0.5, 0.5, 0.5],
        sequence_history=[],
        most_frequent_country="US",
        known_mac_addresses=[],
        known_os_profiles=[],
        known_protocols=[],
        resource_access_counts={},
        command_frequency={},
        event_count=20,
        cold_start_flag=False,
        last_updated="2026-07-25T00:00:00+00:00",
        profile_version=3,
    )


@pytest.fixture
def profile_store(sample_profile: EntityProfile) -> InMemoryProfileStore:
    """Provide an observable store initialized with the sample profile."""
    return InMemoryProfileStore(profile=sample_profile)


def test_update_below_threshold_applies_configured_ewma(
    profile_store: InMemoryProfileStore,
) -> None:
    """A low-risk event updates every baseline dimension with the EWMA formula."""
    config = DriftConfig(ewma_alpha=0.2, gating_threshold=0.4)
    updater = EWMAUpdater(config, profile_store)

    updated = updater.update(
        "usr_001", anomaly_score=0.39, session_features=[6.0, 7.0, 8.0]
    )

    assert updated is not None
    assert updated.baseline_vector == pytest.approx([2.0, 3.0, 4.0])
    assert updated.profile_version == 4
    assert updated.last_updated != "2026-07-25T00:00:00+00:00"
    assert profile_store.upsert_count == 1


@pytest.mark.parametrize("anomaly_score", [0.4, 0.9])
def test_update_at_or_above_threshold_does_not_modify_profile(
    profile_store: InMemoryProfileStore, anomaly_score: float
) -> None:
    """Boundary and high-risk scores cannot be learned into the baseline."""
    original_profile = profile_store.profile
    updater = EWMAUpdater(DriftConfig(gating_threshold=0.4), profile_store)

    updated = updater.update("usr_001", anomaly_score, [6.0, 7.0, 8.0])

    assert updated is None
    assert profile_store.profile == original_profile
    assert profile_store.upsert_count == 0


def test_update_without_existing_profile_does_not_write() -> None:
    """Cold-start entities stay under the cold-start handler's ownership."""
    profile_store = InMemoryProfileStore(profile=None)
    updater = EWMAUpdater(DriftConfig(), profile_store)

    updated = updater.update("usr_new", anomaly_score=0.1, session_features=[1.0, 2.0])

    assert updated is None
    assert profile_store.upsert_count == 0


@pytest.mark.parametrize("anomaly_score", [-0.01, 1.01])
def test_should_update_rejects_scores_outside_normalized_range(
    anomaly_score: float,
) -> None:
    """Invalid fusion outputs are rejected at the module boundary."""
    updater = EWMAUpdater(DriftConfig(), InMemoryProfileStore(profile=None))

    with pytest.raises(ValueError, match="anomaly_score"):
        updater.should_update(anomaly_score)


def test_update_rejects_feature_vector_with_wrong_dimension(
    profile_store: InMemoryProfileStore,
) -> None:
    """The updater protects the profile contract from dimension mismatches."""
    updater = EWMAUpdater(DriftConfig(), profile_store)

    with pytest.raises(ValueError, match="length"):
        updater.update("usr_001", anomaly_score=0.1, session_features=[1.0, 2.0])


def test_drift_config_rejects_invalid_policy_values() -> None:
    """The configurable alpha and gate remain valid normalized values."""
    with pytest.raises(ValueError):
        DriftConfig(ewma_alpha=0.0)
    with pytest.raises(ValueError):
        DriftConfig(gating_threshold=1.1)
