import os
import tempfile
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from anomaly_detection.common.models.features import EntityFeatureVector
from anomaly_detection.common.models.enums import EntityStatus, EntityType
from anomaly_detection.common.models.entities import EntityProfile
from anomaly_detection.feature_engineering.profile_store_interface import ProfileStoreInterface

from src.profiling.config import ProfilingConfig, FEATURE_NAMES
from src.profiling.population_prior import PopulationPrior
from src.profiling.profile_store import ProfileStore
from src.profiling.profile_model import BehavioralProfilingModel, ExtendedProfilingOutput


@pytest.fixture
def config_and_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        profile_path = os.path.join(tmpdir, "profiles", "store.json")
        prior_path = os.path.join(tmpdir, "priors", "population.json")
        config = ProfilingConfig(
            graduation_threshold=5,
            z_score_cap=5.0,
            variance_epsilon=1e-4,
            profile_store_path=profile_path,
            population_prior_path=prior_path
        )
        yield config, profile_path, prior_path


@pytest.fixture
def empty_model(config_and_paths):
    config, _, _ = config_and_paths
    store = ProfileStore(persistence_path=config.profile_store_path)
    prior = PopulationPrior(config=config)
    model = BehavioralProfilingModel(config=config, store=store, prior=prior)
    return model, store, prior


def test_interface_compliance(empty_model):
    """M06's ProfileStore passes M05's ProfileStoreInterface contract tests."""
    _, store, _ = empty_model
    # Structural subtyping check
    assert isinstance(store, ProfileStoreInterface)


def test_fit_builds_profiles_and_prior(empty_model):
    """fit() correctly builds per-entity profiles from training data and prior matches aggregate stats."""
    model, store, prior = empty_model
    
    # Create training dataframe
    data = []
    for _ in range(10):
        # 10 identical events for usr_1 (zero variance case)
        data.append({"entity_id": "usr_1", "entity_type": EntityType.USER, "feature_vector": [1.0] * 24})
        # 10 events for usr_2 with variance
    
    for i in range(10):
        data.append({"entity_id": "usr_2", "entity_type": EntityType.USER, "feature_vector": [float(i)] * 24})

    df = pd.DataFrame(data)
    model.fit(df)

    # Check profiles
    profile1 = store.get_profile("usr_1")
    assert profile1 is not None
    assert profile1.event_count == 10
    # Mean should be 1.0, std should be epsilon (due to clamping)
    assert profile1.baseline_vector[0] == 1.0
    assert profile1.baseline_std[0] == model.config.variance_epsilon
    assert profile1.cold_start_flag is False

    profile2 = store.get_profile("usr_2")
    assert profile2.event_count == 10
    assert profile2.baseline_vector[0] == pytest.approx(4.5)  # mean of 0..9

    # Check prior
    mean_vec, std_vec = prior.get_prior(EntityType.USER)
    assert len(mean_vec) == 24
    assert len(std_vec) == 24
    # The overall mean of twenty items: ten 1s and one 0..9 sequence
    # Sum = 10 + 45 = 55. Mean = 55 / 20 = 2.75
    assert mean_vec[0] == pytest.approx(2.75)


def test_cold_start_scoring(empty_model):
    """Entities not in training set are scored against population prior with entity_status='cold_start'."""
    model, store, prior = empty_model
    
    # We manually set a prior for USER to test scoring
    prior._priors[EntityType.USER.value] = ([2.0] * 24, [1.0] * 24)

    # Score a new entity not in store
    vector = EntityFeatureVector(root=[3.0] * 24)
    output = model.score("new_usr", "evt_1", EntityType.USER, vector)
    
    assert output.entity_status == EntityStatus.COLD_START
    assert output.is_cold_start is True
    assert output.profile_age == 0
    assert output.confidence == 0.6
    
    # Z-score = (3.0 - 2.0) / 1.0 = 1.0
    assert output.per_feature_deviations[FEATURE_NAMES[0]] == 1.0
    # Cap is 5.0, so 1.0/5.0 = 0.2
    assert output.anomaly_score == 0.2


def test_graduation(empty_model):
    """Entities transition to warm status after graduation_threshold upsert() calls."""
    model, store, prior = empty_model
    
    profile = EntityProfile(
        entity_id="usr_graduating",
        entity_type=EntityType.USER,
        baseline_vector=[0.0] * 24,
        baseline_std=[1.0] * 24,
        sequence_history=[],
        most_frequent_country="US",
        known_mac_addresses=[],
        known_os_profiles=[],
        known_protocols=[],
        resource_access_counts={},
        command_frequency={},
        event_count=4, # Threshold is 5
        cold_start_flag=True,
        last_updated=datetime.now(timezone.utc).isoformat(),
        profile_version=1
    )
    store.upsert_profile(profile)

    prior._priors[EntityType.USER.value] = ([10.0] * 24, [1.0] * 24)
    vector = EntityFeatureVector(root=[0.0] * 24)
    
    # At event_count 4, it should be cold start and scored against prior (mean 10)
    output1 = model.score("usr_graduating", "evt_1", EntityType.USER, vector)
    assert output1.is_cold_start is True
    assert output1.entity_status == EntityStatus.COLD_START
    # Z = (0 - 10) / 1 = -10. Max abs = 10. Normalised = 10/5 = 1.0 (capped)
    assert output1.anomaly_score == 1.0

    # Upsert with event_count 5
    profile.event_count = 5
    profile.cold_start_flag = False
    profile.profile_version = 2
    store.upsert_profile(profile)

    # Now it should be warm and scored against profile (mean 0)
    output2 = model.score("usr_graduating", "evt_2", EntityType.USER, vector)
    assert output2.is_cold_start is False
    assert output2.entity_status == EntityStatus.WARM
    assert output2.confidence == 1.0
    # Z = (0 - 0) / 1 = 0
    assert output2.anomaly_score == 0.0


def test_deviation_scoring_and_per_feature(empty_model):
    """Known-anomalous vectors score higher. per_feature_deviations present and signed."""
    model, store, prior = empty_model
    
    profile = EntityProfile(
        entity_id="usr_1",
        entity_type=EntityType.USER,
        baseline_vector=[0.0] * 24,
        baseline_std=[2.0] * 24,
        sequence_history=[],
        most_frequent_country="US",
        known_mac_addresses=[],
        known_os_profiles=[],
        known_protocols=[],
        resource_access_counts={},
        command_frequency={},
        event_count=10,
        cold_start_flag=False,
        last_updated=datetime.now(timezone.utc).isoformat(),
        profile_version=1
    )
    store.upsert_profile(profile)

    # Normal vector
    normal_vec = EntityFeatureVector(root=[1.0] * 24)
    normal_out = model.score("usr_1", "e1", EntityType.USER, normal_vec)
    
    # Anomalous vector (e.g. login at 3am, value=10.0)
    anomalous_vec = EntityFeatureVector(root=[10.0] * 24)
    anomalous_out = model.score("usr_1", "e2", EntityType.USER, anomalous_vec)
    
    assert anomalous_out.anomaly_score > normal_out.anomaly_score
    
    # Z-scores present for all features
    assert len(anomalous_out.per_feature_deviations) == 24
    
    # Signed properly: (10 - 0) / 2 = 5.0
    assert anomalous_out.per_feature_deviations[FEATURE_NAMES[0]] == 5.0


def test_persistence_roundtrip(config_and_paths):
    """save/load round-trip produces identical outputs."""
    config, profile_path, prior_path = config_and_paths
    
    store1 = ProfileStore(persistence_path=profile_path)
    prior1 = PopulationPrior(config=config)
    model1 = BehavioralProfilingModel(config=config, store=store1, prior=prior1)
    
    # Setup state
    profile = EntityProfile(
        entity_id="usr_save",
        entity_type=EntityType.USER,
        baseline_vector=[1.0] * 24,
        baseline_std=[1.0] * 24,
        sequence_history=[],
        most_frequent_country="US",
        known_mac_addresses=[],
        known_os_profiles=[],
        known_protocols=[],
        resource_access_counts={},
        command_frequency={},
        event_count=20,
        cold_start_flag=False,
        last_updated=datetime.now(timezone.utc).isoformat(),
        profile_version=1
    )
    store1.upsert_profile(profile)
    store1.save()
    
    prior1._priors[EntityType.USER.value] = ([2.0] * 24, [2.0] * 24)
    prior1.save()

    # Load in new instances
    store2 = ProfileStore(persistence_path=profile_path)
    store2.load()
    prior2 = PopulationPrior(config=config)
    prior2.load()
    model2 = BehavioralProfilingModel(config=config, store=store2, prior=prior2)
    
    # Score with both, they should be identical
    vec = EntityFeatureVector(root=[5.0] * 24)
    out1 = model1.score("usr_save", "e1", EntityType.USER, vec)
    out2 = model2.score("usr_save", "e1", EntityType.USER, vec)
    
    assert out1.anomaly_score == out2.anomaly_score
    assert out1.per_feature_deviations == out2.per_feature_deviations


def test_edge_case_single_event(empty_model):
    """Entity with exactly one event."""
    model, store, prior = empty_model
    
    data = [{"entity_id": "usr_single", "entity_type": EntityType.USER, "feature_vector": [4.0] * 24}]
    model.fit(pd.DataFrame(data))
    
    profile = store.get_profile("usr_single")
    assert profile is not None
    assert profile.event_count == 1
    # std should be clamped to epsilon since 1 sample has 0 variance
    assert profile.baseline_std[0] == model.config.variance_epsilon

    # The prior will also have 1 item, so mean=4, std=epsilon
    vec = EntityFeatureVector(root=[4.0] * 24)
    out = model.score("usr_single", "e1", EntityType.USER, vec)
    # Z-score = 0
    assert out.anomaly_score == 0.0
